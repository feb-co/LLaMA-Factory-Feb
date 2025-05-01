#
# Created on Thur May 01 2025
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
from typing import TYPE_CHECKING, Any, Optional


from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..data_utils import Role
from .processor_utils import DatasetProcessor, packing_conversation


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class ConversationDatasetProcessor(DatasetProcessor):
    def _encode_normal_message(
        self,
        messages: Optional[list],
        train_on_prompt: bool,
        mask_history: bool,
        input_ids: Optional[list],
        labels: Optional[list],
    ):
        encoded_pairs = self.template.encode_multiturn(tokenizer=self.tokenizer, messages=messages, system=None, tools=None)
        text_pairs = [(messages[i], messages[i + 1]) for i in range(0, len(messages), 2)]
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            source_len = len(source_ids)
            target_len = len(target_ids)

            if train_on_prompt:
                source_label = source_ids
            elif turn_idx != 0 and self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
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
        self,
        messages: Optional[list],
        train_on_prompt: bool,
        mask_history: bool,
        input_ids: Optional[list],
        labels: Optional[list],
    ):
        encoded_pairs = self.template.encode_multiturn_with_longthought(tokenizer=self.tokenizer, messages=messages, system=None, tools=None)
        text_pairs = [(messages[i], messages[i + 1], messages[i + 2]) for i in range(0, len(messages), 3)]
        for turn_idx, (source_ids, thought_ids, target_ids) in enumerate(encoded_pairs):
            source_len = len(source_ids)
            thought_len = len(thought_ids)
            target_len = len(target_ids)

            if train_on_prompt:
                source_label = source_ids
            elif turn_idx != 0 and self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
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

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> tuple[list[int], list[int], list[int]]:
        messages = prompt + response
        input_ids, labels = [], []

        prefix_ids = self.template.encode_system(tokenizer=self.tokenizer, system=system, tools=tools)
        if len(messages)%2 == 0:
            input_ids, labels = self._encode_normal_message(messages, self.data_args.train_on_prompt, self.data_args.mask_history, input_ids, labels)
        elif len(messages)%3 == 0:
            input_ids, labels = self._encode_longthought_message(messages, self.data_args.train_on_prompt, self.data_args.mask_history, input_ids, labels)
        else:
            raise NotImplementedError

        assert len(input_ids) == len(labels), "The length of input_ids should equal with labels' length!"
        return prefix_ids, input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(examples["_prompt"])):
            system_ids, input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
            )
            model_inputs["input_ids"].append(system_ids + input_ids)
            model_inputs["attention_mask"].append([1] * (len(system_ids) + len(input_ids)))
            model_inputs["labels"].append([IGNORE_INDEX] * len(system_ids) + labels)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]), flush=True)
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)), flush=True)
        print("label_ids:\n{}".format(example["labels"]), flush=True)
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}", flush=True)


@dataclass
class PackedConversationDatasetProcessor(ConversationDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        batch_input_ids, batch_labels = defaultdict(list), defaultdict(list)
        lengths = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            system_ids, input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
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
                self.data_args.cutoff_len
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
                if len(packed_input_ids) < self.data_args.cutoff_len:
                    pad_length = self.data_args.cutoff_len - len(packed_input_ids)
                    packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                    packed_labels += [IGNORE_INDEX] * pad_length
                    packed_attention_masks += [1] * pad_length
                elif len(packed_input_ids) > self.data_args.cutoff_len:
                    packed_input_ids = packed_input_ids[: self.data_args.cutoff_len]
                    packed_attention_masks = packed_attention_masks[: self.data_args.cutoff_len]
                    packed_labels = packed_labels[: self.data_args.cutoff_len]

                assert (
                    len(packed_input_ids)
                    == len(packed_attention_masks)
                    == len(packed_labels)
                    == self.data_args.cutoff_len
                ), "The length of packed example should be identical to the cutoff length."

                model_inputs["input_ids"].append(packed_input_ids)
                model_inputs["attention_mask"].append(packed_attention_masks)
                model_inputs["labels"].append(packed_labels)

        return model_inputs


@dataclass
class InstructionDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> tuple[list[int], list[int], list[int]]:
        messages = prompt + response
        input_ids, labels = [], []
        prefix_ids = self.template.encode_system(tokenizer=self.tokenizer, system=system, tools=tools)
        encoded_pairs = self.template.encode_instruction(tokenizer=self.tokenizer, messages=messages)
        text_pairs = [(messages[i], messages[i + 1]) for i in range(0, len(messages), 2)]
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            source_len = len(source_ids)
            target_len = len(target_ids)

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif turn_idx != 0 and self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != len(encoded_pairs) - 1:
                target_label = [IGNORE_INDEX] * target_len
            elif text_pairs[turn_idx][1]["role"] == Role.MASK.value:
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            input_ids += source_ids + target_ids
            labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        assert len(input_ids) == len(
            labels
        ), "The length of input_ids should equal with labels' length!"

        return prefix_ids, input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(examples["_prompt"])):
            prefix_ids, input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
            )

            concat_input_ids = prefix_ids + input_ids
            concat_labels = [IGNORE_INDEX] * len(prefix_ids) + labels

            if len(concat_input_ids) < self.data_args.cutoff_len:
                pad_length = self.data_args.cutoff_len - len(concat_input_ids)
                concat_input_ids += [self.tokenizer.pad_token_id] * pad_length
                concat_labels += [IGNORE_INDEX] * pad_length
            elif len(concat_input_ids) > self.data_args.cutoff_len:
                concat_input_ids = concat_input_ids[: self.data_args.cutoff_len]
                concat_labels = concat_labels[: self.data_args.cutoff_len]

            attention_masks = [1] * len(concat_input_ids)

            assert (
                len(concat_input_ids)
                == len(attention_masks)
                == len(concat_labels)
                == self.data_args.cutoff_len
            ), "The length of packed example should be identical to the cutoff length."

            model_inputs["input_ids"].append(concat_input_ids)
            model_inputs["attention_mask"].append(attention_masks)
            model_inputs["labels"].append(concat_labels)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]), flush=True)
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)), flush=True)
        print("label_ids:\n{}".format(example["labels"]), flush=True)
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}", flush=True)
