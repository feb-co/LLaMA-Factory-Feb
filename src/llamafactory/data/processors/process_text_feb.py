#
# Created on Sat Aug 03 2024
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


from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import (
    get_paligemma_token_type_ids,
    get_pixel_values,
    packing_conversation,
)


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)




"""
conversation
"""
def _encode_conversation_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
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
    encoded_pairs = template.encode_multiturn(
        tokenizer=tokenizer, messages=messages, system=None, tools=None
    )
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if train_on_prompt:
            source_label = source_ids
        else:
            source_label = [IGNORE_INDEX] * len(source_ids)

        if mask_history and turn_idx != len(encoded_pairs) - 1:
            target_label = [IGNORE_INDEX] * len(target_ids)
        else:
            target_label = target_ids[1:] + [tokenizer.eos_token_id]

        input_ids += source_ids + target_ids
        labels += source_label + target_label

    input_ids += [tokenizer.eos_token_id]
    labels += [IGNORE_INDEX]

    return prefix_ids, input_ids, labels


def preprocess_conversation_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning(
                "Dropped invalid example: {}".format(
                    examples["prompt"][i] + examples["response"][i]
                )
            )
            continue

        system_ids, input_ids, labels = _encode_conversation_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        model_inputs["input_ids"].append(system_ids + input_ids)
        model_inputs["attention_mask"].append([1] * (len(system_ids) + len(input_ids)))
        model_inputs["labels"].append([IGNORE_INDEX] * len(system_ids) + labels)
        if processor is not None:
            model_inputs["pixel_values"].append(
                get_pixel_values(examples["images"][i], processor)
            )
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["token_type_ids"].append(
                    get_paligemma_token_type_ids(len(input_ids), processor)
                )

    return model_inputs


def preprocess_packed_conversation_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    batch_input_ids, batch_labels = defaultdict(list), defaultdict(list)
    lengths = defaultdict(list)
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning(
                "Dropped invalid example: {}".format(
                    examples["prompt"][i] + examples["response"][i]
                )
            )
            continue

        system_ids, input_ids, labels = _encode_conversation_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=None,
            cutoff_len=data_args.cutoff_len - 1,  # reserved for the padding token
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
        sub_model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
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
                if data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(
                        batch_input_ids[system_ids_key][index]
                    )  # start from 1
                else:
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
                if data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length
            elif len(packed_input_ids) > data_args.cutoff_len:
                packed_input_ids = packed_input_ids[: data_args.cutoff_len]
                packed_attention_masks = packed_attention_masks[: data_args.cutoff_len]
                packed_labels = packed_labels[: data_args.cutoff_len]

            assert (
                len(packed_input_ids)
                == len(packed_attention_masks)
                == len(packed_labels)
                == len(data_args.cutoff_len)
            ), "The length of packed example should be identical to the cutoff length."

            sub_model_inputs["input_ids"].append(packed_input_ids)
            sub_model_inputs["attention_mask"].append(packed_attention_masks)
            sub_model_inputs["labels"].append(packed_labels)

        model_inputs["input_ids"] += sub_model_inputs["input_ids"]
        model_inputs["attention_mask"] += sub_model_inputs["attention_mask"]
        model_inputs["labels"] += sub_model_inputs["labels"]

    return model_inputs


def print_conversation_dataset_example(
    example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer"
) -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print(
        "inputs:\n{}".format(
            tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        )
    )
    print("label_ids:\n{}".format(example["labels"]))
    print(
        "labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False))
    )


"""
instruction
"""
def _encode_instruction_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    template: "Template",
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

    encoded_pairs = template.encode_instruction(
        tokenizer=tokenizer, messages=messages
    )
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if train_on_prompt:
            source_label = source_ids
        else:
            source_label = [IGNORE_INDEX] * len(source_ids)

        if mask_history and turn_idx != len(encoded_pairs) - 1:
            target_label = [IGNORE_INDEX] * len(target_ids)
        else:
            target_label = target_ids[1:] + [tokenizer.eos_token_id]

        input_ids += source_ids + target_ids
        labels += source_label + target_label
    
    return input_ids, labels


def preprocess_instruction_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        input_ids, labels = _encode_instruction_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        
        if len(input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_length
            labels += [IGNORE_INDEX] * pad_length
        elif len(input_ids) > data_args.cutoff_len:
            input_ids = input_ids[: data_args.cutoff_len]
            labels = labels[: data_args.cutoff_len]
        
        attention_masks = [1] * len(input_ids)
        
        assert (
            len(input_ids)
            == len(attention_masks)
            == len(labels)
            == len(data_args.cutoff_len)
        ), "The length of packed example should be identical to the cutoff length."

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_masks)
        model_inputs["labels"].append(labels)
        
        if processor is not None:
            model_inputs["pixel_values"].append(
                get_pixel_values(examples["images"][i], processor)
            )
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["token_type_ids"].append(
                    get_paligemma_token_type_ids(len(input_ids), processor)
                )

    return model_inputs


"""
pretrain
"""
def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[List[int]]]:
    eos_token = tokenizer.eos_token
    text_examples = [tokenizer.bos_token + messages[0]["content"] + eos_token for messages in examples["prompt"]]

    if not data_args.packing:
        model_inputs = tokenizer(text_examples, add_special_tokens=False, max_length=data_args.cutoff_len, truncation=True)
    else:
        batch_input_ids, batch_labels = [], []
        lengths = []
        for text in text_examples:
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            labels = input_ids[1:]
            input_ids = input_ids[:-1]
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
                packed_attention_masks += [1] * len(batch_input_ids[index])
            
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
                == len(data_args.cutoff_len)
            ), "The length of packed example should be identical to the cutoff length."

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["labels"].append(packed_labels)

    return model_inputs