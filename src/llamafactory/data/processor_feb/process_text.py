#
# Created on Sat Aug 03 2024
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple


from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..data_utils import Role
from .processor_utils import packing_conversation


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template_feb import TemplateFeb


logger = logging.get_logger(__name__)


# """
# conversation
# """
def _encode_normal_message(
    messages: Optional[list],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    train_on_prompt: bool,
    mask_history: bool,
    input_ids: Optional[list],
    labels: Optional[list],
):
    encoded_pairs = template.encode_multiturn(tokenizer=tokenizer, messages=messages, system=None, tools=None)
    text_pairs = [(messages[i], messages[i + 1]) for i in range(0, len(messages), 2)]
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        source_len = len(source_ids)
        target_len = len(target_ids)

        if train_on_prompt:
            source_label = source_ids
        elif turn_idx != 0 and template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != len(encoded_pairs) - 1:
            target_label = [IGNORE_INDEX] * target_len
        elif text_pairs[turn_idx][1]["role"] == Role.MASK.value:
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        input_ids += source_ids + target_ids
        labels += source_label + target_label

    return input_ids, labels


def _encode_longthought_message(
    messages: Optional[list],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    train_on_prompt: bool,
    mask_history: bool,
    input_ids: Optional[list],
    labels: Optional[list],
):
    encoded_pairs = template.encode_multiturn_with_longthought(tokenizer=tokenizer, messages=messages, system=None, tools=None)
    text_pairs = [(messages[i], messages[i + 1], messages[i + 2]) for i in range(0, len(messages), 3)]
    for turn_idx, (source_ids, thought_ids, target_ids) in enumerate(encoded_pairs):
        source_len = len(source_ids)
        thought_len = len(thought_ids)
        target_len = len(target_ids)

        if train_on_prompt:
            source_label = source_ids
        elif turn_idx != 0 and template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != len(encoded_pairs) - 1:
            thought_label = [IGNORE_INDEX] * thought_len
        else:
            thought_label = thought_ids

        if mask_history and turn_idx != len(encoded_pairs) - 1:
            target_label = [IGNORE_INDEX] * target_len
        elif text_pairs[turn_idx][2]["role"] == Role.MASK.value:
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        input_ids += source_ids + thought_ids + target_ids
        labels += source_label + thought_label + target_label

    return input_ids, labels


def _encode_conversation_example(
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
    input_ids, labels = [], []

    if processor is not None and hasattr(
        processor, "image_seq_length"
    ):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        input_ids += [image_token_id] * getattr(processor, "image_seq_length")
        labels += [IGNORE_INDEX] * getattr(processor, "image_seq_length")

    prefix_ids = template.encode_system(tokenizer=tokenizer, system=system, tools=tools)
    if len(messages)%2 == 0:
        input_ids, labels = _encode_normal_message(messages, template, tokenizer, train_on_prompt, mask_history, input_ids, labels)
    elif len(messages)%3 == 0:
        input_ids, labels = _encode_longthought_message(messages, template, tokenizer, train_on_prompt, mask_history, input_ids, labels)
    else:
        raise NotImplementedError

    assert len(input_ids) == len(labels), "The length of input_ids should equal with labels' length!"
    return prefix_ids, input_ids, labels


def preprocess_conversation_dataset(
    examples: Dict[str, List[Any]],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["_prompt"])):
        system_ids, input_ids, labels = _encode_conversation_example(
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
        model_inputs["input_ids"].append(system_ids + input_ids)
        model_inputs["attention_mask"].append([1] * (len(system_ids) + len(input_ids)))
        model_inputs["labels"].append([IGNORE_INDEX] * len(system_ids) + labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def preprocess_packed_conversation_dataset(
    examples: Dict[str, List[Any]],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # TODO: use `position_ids` to achieve packing
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    batch_input_ids, batch_labels = defaultdict(list), defaultdict(list)
    lengths = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        system_ids, input_ids, labels = _encode_conversation_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=None,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        length = len(input_ids)

        system_ids_key = tuple(system_ids)
        if system_ids_key not in lengths:
            lengths[system_ids_key] = []
            batch_input_ids[system_ids_key] = []
            batch_labels[system_ids_key] = []

        lengths[system_ids_key].append(length)
        batch_input_ids[system_ids_key].append(input_ids)
        batch_labels[system_ids_key].append(labels)

    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for system_ids_key in lengths.keys():
        system_ids = list(system_ids_key)
        system_ids_len = len(system_ids)
        knapsacks = packing_conversation(
            lengths[system_ids_key],
            data_args.cutoff_len
            - system_ids_len,  # reserved for the padding token and system prompt
        )
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_labels = [], [], []
            for i, index in enumerate(knapsack):
                packed_input_ids += batch_input_ids[system_ids_key][index]
                packed_labels += batch_labels[system_ids_key][index]
                packed_attention_masks += [1] * len(
                    batch_input_ids[system_ids_key][index]
                )

            packed_input_ids = system_ids + packed_input_ids
            packed_attention_masks = [1] * system_ids_len + packed_attention_masks
            packed_labels = [IGNORE_INDEX] * system_ids_len + packed_labels
            if len(packed_input_ids) < data_args.cutoff_len:
                pad_length = data_args.cutoff_len - len(packed_input_ids)
                packed_input_ids += [tokenizer.pad_token_id] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                packed_attention_masks += [1] * pad_length
            elif len(packed_input_ids) > data_args.cutoff_len:
                packed_input_ids = packed_input_ids[: data_args.cutoff_len]
                packed_attention_masks = packed_attention_masks[: data_args.cutoff_len]
                packed_labels = packed_labels[: data_args.cutoff_len]

            assert (
                len(packed_input_ids)
                == len(packed_attention_masks)
                == len(packed_labels)
                == data_args.cutoff_len
            ), "The length of packed example should be identical to the cutoff length."

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["labels"].append(packed_labels)

    return model_inputs


def print_conversation_dataset_example(
    example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer"
) -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]), flush=True)
    print(
        "inputs:\n{}".format(
            tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        ),
        flush=True,
    )
    print("label_ids:\n{}".format(example["labels"]), flush=True)
    print(
        "labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)),
        flush=True,
    )


# """
# instruction
# """
def _encode_instruction_example(
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
    input_ids, labels = [], []

    if processor is not None and hasattr(
        processor, "image_seq_length"
    ):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        input_ids += [image_token_id] * getattr(processor, "image_seq_length")
        labels += [IGNORE_INDEX] * getattr(processor, "image_seq_length")

    prefix_ids = template.encode_system(tokenizer=tokenizer, system=system, tools=tools)
    encoded_pairs = template.encode_instruction(tokenizer=tokenizer, messages=messages)
    text_pairs = [(messages[i], messages[i + 1]) for i in range(0, len(messages), 2)]
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        source_len = len(source_ids)
        target_len = len(target_ids)

        if train_on_prompt:
            source_label = source_ids
        elif turn_idx != 0 and template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != len(encoded_pairs) - 1:
            target_label = [IGNORE_INDEX] * target_len
        elif text_pairs[turn_idx][1]["role"] == Role.MASK.value:
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        input_ids += source_ids + target_ids
        labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    assert len(input_ids) == len(
        labels
    ), "The length of input_ids should equal with labels' length!"

    return prefix_ids, input_ids, labels


def preprocess_instruction_dataset(
    examples: Dict[str, List[Any]],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["_prompt"])):
        prefix_ids, input_ids, labels = _encode_instruction_example(
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

        concat_input_ids = prefix_ids + input_ids
        concat_labels = [IGNORE_INDEX] * len(prefix_ids) + labels

        if len(concat_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(concat_input_ids)
            concat_input_ids += [tokenizer.pad_token_id] * pad_length
            concat_labels += [IGNORE_INDEX] * pad_length
        elif len(concat_input_ids) > data_args.cutoff_len:
            concat_input_ids = concat_input_ids[: data_args.cutoff_len]
            concat_labels = concat_labels[: data_args.cutoff_len]

        attention_masks = [1] * len(concat_input_ids)

        assert (
            len(concat_input_ids)
            == len(attention_masks)
            == len(concat_labels)
            == data_args.cutoff_len
        ), "The length of packed example should be identical to the cutoff length."

        model_inputs["input_ids"].append(concat_input_ids)
        model_inputs["attention_mask"].append(attention_masks)
        model_inputs["labels"].append(concat_labels)

    return model_inputs


# """
# pretrain
# """
def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    eos_token = tokenizer.eos_token
    text_examples = [
        tokenizer.bos_token + messages[0]["content"] + eos_token
        for messages in examples["_prompt"]
    ]

    if not data_args.packing:
        model_inputs = tokenizer(
            text_examples,
            add_special_tokens=False,
            max_length=data_args.cutoff_len,
            truncation=True,
        )
    else:
        batch_input_ids, batch_labels = [], []
        lengths = []
        for text in text_examples:
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            labels = input_ids[:]
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            lengths.append(len(input_ids))

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        knapsacks = packing_conversation(lengths, data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_labels = [], [], []
            for i, index in enumerate(knapsack):
                packed_input_ids += batch_input_ids[index]
                packed_labels += batch_labels[index]
                if data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(
                        batch_input_ids[index]
                    )  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < data_args.cutoff_len:
                pad_length = data_args.cutoff_len - len(packed_input_ids)
                packed_input_ids += [tokenizer.pad_token_id] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [
                        1
                    ] * pad_length  # more efficient flash_attn
            elif len(packed_input_ids) > data_args.cutoff_len:
                packed_input_ids = packed_input_ids[: data_args.cutoff_len]
                packed_attention_masks = packed_attention_masks[: data_args.cutoff_len]
                packed_labels = packed_labels[: data_args.cutoff_len]

            assert (
                len(packed_input_ids)
                == len(packed_attention_masks)
                == len(packed_labels)
                == data_args.cutoff_len
            ), "The length of packed example should be identical to the cutoff length."

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["labels"].append(packed_labels)

    return model_inputs


def print_pretrain_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))



# """
# DPO
# """
def _encode_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    if processor is not None and not hasattr(
        processor, "image_seq_length"
    ):  # llava-like models
        prompt[0]["content"] = template.image_token + prompt[0]["content"]

    chosen_messages = prompt + [response[0]]
    rejected_messages = prompt + [response[1]]
    prompt_ids, chosen_ids = template.encode_oneturn(
        tokenizer, chosen_messages, system, tools
    )
    _, rejected_ids = template.encode_oneturn(
        tokenizer, rejected_messages, system, tools
    )

    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    if processor is not None and hasattr(
        processor, "image_seq_length"
    ):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        prompt_ids = [image_token_id] * getattr(
            processor, "image_seq_length"
        ) + prompt_ids

    chosen_input_ids = prompt_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids
    rejected_input_ids = prompt_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * len(prompt_ids) + rejected_ids

    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": [],
        "rejected_labels": [],
        "images": [],
        "videos": [],
    }
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            logger.warning(
                "Dropped invalid example: {}".format(
                    examples["_prompt"][i] + examples["_response"][i]
                )
            )
            continue

        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = (
            _encode_pairwise_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                template=template,
                tokenizer=tokenizer,
                processor=processor,
            )
        )

        assert len(chosen_input_ids) == len(chosen_labels)
        assert len(rejected_input_ids) == len(rejected_labels)

        if len(chosen_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(chosen_input_ids)
            chosen_input_ids += [tokenizer.pad_token_id] * pad_length
            chosen_labels += [IGNORE_INDEX] * pad_length
        elif len(chosen_input_ids) > data_args.cutoff_len:
            continue

        if len(rejected_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(rejected_input_ids)
            rejected_input_ids += [tokenizer.pad_token_id] * pad_length
            rejected_labels += [IGNORE_INDEX] * pad_length
        elif len(rejected_input_ids) > data_args.cutoff_len:
            continue

        assert (
            len(chosen_input_ids)
            == len(chosen_labels)
            == len(rejected_input_ids)
            == len(rejected_labels)
            == data_args.cutoff_len
        )

        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)

        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
    valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
    print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
    print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
    print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
    print("chosen_labels:\n{}".format(tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)))
    print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
    print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)))
    print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
    print("rejected_labels:\n{}".format(tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)))

