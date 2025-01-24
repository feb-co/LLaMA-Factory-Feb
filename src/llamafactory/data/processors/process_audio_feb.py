#
# Created on Sat Jan 01 2024
#
# Licheng Wang (FEB team)
#
# The MIT License (MIT)
# Copyright (c) 2024 Licheng Wang (FEB team)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple


from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..data_utils import Role


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template_feb import TemplateFeb


logger = logging.get_logger(__name__)


# """
# avater_audio
# """
def _encode_avater_audio_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    if processor is not None and not hasattr(
        processor, "image_seq_length"
    ):  # llava-like models
        prompt[0]["content"] = template.image_token + prompt[0]["content"]

    messages = prompt + response

    text_input_ids, text_labels, valid_tokens_pos = [], [], []
    audio_features, audio_pos = [], []
    audio_codes_ids, audio_codes_labels = [], []
    t2a_attention_mask = []

    prefix_ids = template.encode_system(tokenizer=tokenizer, system=system, tools=tools)
    encoded_pairs = template.encode_avater_audio(tokenizer=tokenizer, prompt_messages=prompt, response_message=response[-1])
    text_pairs = [(messages[i], messages[i + 1]) for i in range(0, len(messages), 2)]
    for turn_idx, (source_dict, target_dict) in enumerate(encoded_pairs):
        # text
        source_token_ids = source_dict["token_ids"]
        target_token_ids = target_dict["token_ids"]

        source_text_len = len(source_token_ids)
        target_text_len = len(target_token_ids)

        source_text_label = [IGNORE_INDEX] * source_text_len

        if turn_idx != len(encoded_pairs) - 1:
            target_text_label = [IGNORE_INDEX] * target_text_len
        elif text_pairs[turn_idx][1]["role"] == Role.MASK.value:
            target_text_label = [IGNORE_INDEX] * target_text_len
        else:
            target_text_label = target_token_ids[:]
            valid_tokens_pos = [idx for idx in range(
                len(text_labels)+source_text_len, len(text_labels)+source_text_len+len(target_token_ids)
            )]

        text_input_ids += source_token_ids + target_token_ids
        text_labels += source_text_label + target_text_label

        # audio input
        if "audio_features" in source_dict:
            audio_features += source_dict["audio_features"]
            audio_pos += source_dict["audio_pos"]

        # audio output
        if turn_idx == len(encoded_pairs) - 1 and "audio_codes" in target_dict:
            audio_codes_ids = [
                tokenizer.audio_code_shift(target_dict["audio_codes"][idx], layer_idx=idx)
                for idx in range(len(target_dict["audio_codes"]))
            ]
            audio_codes_labels = copy.deepcopy(target_dict["audio_codes"])
            for idx in range(len(audio_codes_labels)):
                if idx == 0:
                    audio_codes_labels[idx] = audio_codes_labels[0][:-tokenizer.acoustic_delay] + [IGNORE_INDEX] * tokenizer.acoustic_delay
                else:
                    audio_codes_labels[idx] = [IGNORE_INDEX] * (tokenizer.acoustic_delay+1) + audio_codes_labels[idx][tokenizer.acoustic_delay+1:]
            t2a_attention_mask = tokenizer.convert_t2a_attention_mask(target_token_ids, target_dict["audio_codes"])

    assert len(text_input_ids) == len(text_labels), "The length of text_input_ids should equal with labels' length!"
    assert (len(audio_codes_ids) == len(audio_codes_labels) and len(audio_codes_ids[0]) == len(audio_codes_labels[0])), "The length of audio_codes_ids should equal with labels' length!"
    return {
        "prefix_ids": prefix_ids,
        "text_input_ids": text_input_ids, "text_labels": text_labels, "valid_tokens_pos": valid_tokens_pos,
        "audio_features": audio_features, "audio_pos": audio_pos,
        "audio_codes_ids": audio_codes_ids, "audio_codes_labels": audio_codes_labels,
        "t2a_attention_mask": t2a_attention_mask,
    }


def preprocess_avater_audio_dataset(
    examples: Dict[str, List[Any]],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {
        "input_ids": [], "attention_mask": [], "labels": [], "valid_tokens_pos": [],
        "decoder_input_ids": [], "decoder_attention_mask": [], "encoder_decoder_attention_mask": [], "decoder_labels": [],
        "images": [], "videos": []
    }
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        enocde_outputs: dict = _encode_avater_audio_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )

        input_ids = enocde_outputs["prefix_ids"] + enocde_outputs["text_input_ids"]
        labels = [IGNORE_INDEX] * len(enocde_outputs["prefix_ids"]) + enocde_outputs["text_labels"]

        if len(input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(input_ids)
            input_ids += [tokenizer.text_tokenizer.pad_token_id] * pad_length
            labels += [IGNORE_INDEX] * pad_length
        elif len(input_ids) > data_args.cutoff_len:
            continue

        # text encoder
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["valid_tokens_pos"].append(enocde_outputs["valid_tokens_pos"])

        # audio encoder
        # TODO

        # tts adapter
        model_inputs["decoder_input_ids"].append(enocde_outputs["audio_codes_ids"])
        model_inputs["decoder_attention_mask"].append([1] * len(enocde_outputs["audio_codes_ids"][0]))
        model_inputs["encoder_decoder_attention_mask"].append(enocde_outputs["t2a_attention_mask"])
        model_inputs["decoder_labels"].append(enocde_outputs["audio_codes_labels"])

        # other
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def print_avater_audio_dataset_example(
    example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer"
) -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]), flush=True)
    print(
        "inputs:\n{}".format(
            tokenizer.text_tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        ),
        flush=True,
    )
    print("label_ids:\n{}".format(example["labels"]), flush=True)
    print(
        "labels:\n{}".format(tokenizer.text_tokenizer.decode(valid_labels, skip_special_tokens=False)),
        flush=True,
    )

