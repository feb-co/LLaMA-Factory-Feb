#
# Created on Sat Jan 01 2024
#
# Licheng Wang (FEB team)
#
# The MIT License (MIT)
# Copyright (c) 2025 Licheng Wang (FEB team)
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

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple


from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import AudioExample, DatasetProcessor, packing_conversation, process_audio_messages


if TYPE_CHECKING:
    from ...hparams import DataArguments


logger = logging.get_logger(__name__)



def _prepare_model_inputs(cutoff_len: int, pad_token_id: int, enocde_outputs: AudioExample, model_inputs: dict):
    input_ids = enocde_outputs["prefix_ids"] + enocde_outputs["text_input_ids"]
    text_labels = [IGNORE_INDEX] * len(enocde_outputs["prefix_ids"]) + enocde_outputs["text_labels"]
    valid_tokens_pos = [pos+len(enocde_outputs["prefix_ids"]) for pos in enocde_outputs["valid_tokens_pos"]]

    if len(input_ids) < cutoff_len:
        pad_length = cutoff_len - len(input_ids)
        input_ids += [pad_token_id] * pad_length
        text_labels += [IGNORE_INDEX] * pad_length
    elif len(input_ids) > cutoff_len and not enocde_outputs["audio_codes_ids"]:
        input_ids = input_ids[:cutoff_len]
        text_labels = text_labels[:cutoff_len]
    elif len(input_ids) > cutoff_len:
        return model_inputs

    # text encoder
    model_inputs["input_ids"].append(input_ids)
    model_inputs["attention_mask"].append([1] * len(input_ids))
    model_inputs["text_labels"].append(text_labels)

    # audio encoder
    model_inputs["audio_features"].append(enocde_outputs["audio_features"] if enocde_outputs["audio_features"] else None)
    model_inputs["audio_positions"].append([[audio_start_p+len(enocde_outputs["prefix_ids"]), audio_len] for audio_start_p, audio_len in enocde_outputs["audio_positions"]] if enocde_outputs["audio_positions"] else None)

    # tts adapter
    model_inputs["valid_tokens_pos"].append(valid_tokens_pos if valid_tokens_pos else None)
    model_inputs["decoder_input_ids"].append(enocde_outputs["audio_codes_ids"] if enocde_outputs["audio_codes_ids"] else None)
    model_inputs["decoder_attention_mask"].append(([1] * len(enocde_outputs["audio_codes_ids"][0])) if enocde_outputs["audio_codes_ids"] else None)
    model_inputs["encoder_decoder_attention_mask"].append(enocde_outputs["t2a_attention_mask"] if enocde_outputs["t2a_attention_mask"] else None)
    model_inputs["decoder_labels"].append(enocde_outputs["audio_codes_labels"] if enocde_outputs["audio_codes_labels"] else None)

    return model_inputs


@dataclass
class AvatarAudioDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: Sequence[dict[str, str]],
        response: Sequence[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> Tuple[list[int], list[int]]:
        if prompt is None or response is None or len(prompt)==0 or len(response)==0:
            return None

        messages = prompt + response

        prefix_ids = self.template.encode_system(tokenizer=self.tokenizer, system=system, tools=tools)

        audio_example: AudioExample = process_audio_messages(
            messages=messages,
            template=self.template,
            tokenizer=self.tokenizer,
            mask_history=self.data_args.mask_history
        )
        
        if audio_example is None:
            return None

        audio_example["prefix_ids"] = prefix_ids
        return audio_example

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {
            "input_ids": [], "attention_mask": [], "text_labels": [],
            "audio_features": [], "audio_positions": [],
            "valid_tokens_pos": [], "encoder_decoder_attention_mask": [],
            "decoder_input_ids": [], "decoder_attention_mask": [], "decoder_labels": [],
        }
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            enocde_outputs: AudioExample = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
            )
            
            if enocde_outputs is None:
                continue

            model_inputs = _prepare_model_inputs(self.data_args.cutoff_len, self.tokenizer.text_tokenizer.pad_token_id, enocde_outputs, model_inputs)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        # text
        text_labels = list(filter(lambda x: x != IGNORE_INDEX, example["text_labels"]))
        print("input_ids:\n{}".format(example["input_ids"]), flush=True)
        print(
            "inputs:\n{}".format(
                self.tokenizer.text_tokenizer.decode(example["input_ids"], skip_special_tokens=False)
            ),
            flush=True,
        )
        print("text_label_ids:\n{}".format(example["text_labels"]), flush=True)
        print(
            "labels:\n{}".format(self.tokenizer.text_tokenizer.decode(text_labels, skip_special_tokens=False)),
            flush=True,
        )


@dataclass
class PackedAvatarAudioDatasetProcessor(AvatarAudioDatasetProcessor):
    def _prepare_packed_example(system_ids_key, knapsack, batch_model_inputs: dict):
        system_ids = list(system_ids_key)
        cur_start_pos = 0
        text_input_ids, text_labels,  = [], []
        audio_features, audio_positions = [], []
        audio_codes_ids, audio_codes_labels = [], []
        valid_tokens_pos, t2a_attention_mask = [], []
        for i, index in enumerate(knapsack):
            # text
            text_input_ids += batch_model_inputs[system_ids_key][index]["text_input_ids"]
            text_labels += batch_model_inputs[system_ids_key][index]["text_labels"]
            
            # audio input
            audio_features += batch_model_inputs[system_ids_key][index]["audio_features"]
            audio_positions += [[cur_start_pos+audio_start_p, audio_len] for audio_start_p, audio_len in batch_model_inputs[index]["audio_features"]]

            # audio output (only for last item valid)
            if i == len(knapsack)-1:
                audio_codes_ids = batch_model_inputs[system_ids_key][index]["audio_codes_ids"]
                audio_codes_labels = batch_model_inputs[system_ids_key][index]["audio_codes_labels"]
                t2a_attention_mask = batch_model_inputs[system_ids_key][index]["t2a_attention_mask"]
                valid_tokens_pos = [cur_start_pos+pos for pos in batch_model_inputs[system_ids_key][index]["valid_tokens_pos"]]

            cur_start_pos += len(batch_model_inputs[index]["text_input_ids"])
        
        assert len(text_input_ids) == len(text_labels), "The length of text_input_ids should equal with labels' length!"
        assert (len(audio_codes_ids) == len(audio_codes_labels) and len(audio_codes_ids[0]) == len(audio_codes_labels[0])), "The length of audio_codes_ids should equal with labels' length!"
        return {
            "prefix_ids": system_ids,
            "text_input_ids": text_input_ids, "text_labels": text_labels,
            "audio_features": audio_features, "audio_positions": audio_positions,
            "audio_codes_ids": audio_codes_ids, "audio_codes_labels": audio_codes_labels,
            "valid_tokens_pos": valid_tokens_pos, "t2a_attention_mask": t2a_attention_mask,
        }

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        lengths, batch_model_inputs = defaultdict(list), defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            enocde_outputs: AudioExample = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
            )

            if enocde_outputs is None:
                continue

            system_ids_key = tuple(enocde_outputs["prefix_ids"])
            if system_ids_key not in lengths:
                lengths[system_ids_key] = []
                batch_model_inputs[system_ids_key] = []

            lengths[system_ids_key].append(len(enocde_outputs["text_input_ids"]))
            batch_model_inputs[system_ids_key].append(enocde_outputs)

        model_inputs = {
            "input_ids": [], "attention_mask": [], "text_labels": [],
            "audio_features": [], "audio_positions": [],
            "valid_tokens_pos": [], "encoder_decoder_attention_mask": [],
            "decoder_input_ids": [], "decoder_attention_mask": [], "decoder_labels": [],
        }
        for system_ids_key in lengths.keys():
            system_ids_len = len(system_ids_key)
            knapsacks = packing_conversation(
                lengths[system_ids_key],
                self.data_args.cutoff_len
                - system_ids_len,  # reserved for the padding token and system prompt
            )
            for knapsack in knapsacks:
                enocde_outputs: dict = self._prepare_packed_example(system_ids_key, knapsack, batch_model_inputs)
                model_inputs = _prepare_model_inputs(self.data_args.cutoff_len, self.tokenizer.text_tokenizer.pad_token_id, enocde_outputs, model_inputs)

        return model_inputs
