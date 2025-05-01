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

from dataclasses import dataclass
from typing import Any

from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, packing_conversation


@dataclass
class PretrainDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        eos_token = self.tokenizer.eos_token
        text_examples = [
            self.tokenizer.bos_token + messages[0]["content"] + eos_token
            for messages in examples["_prompt"]
        ]

        if not self.data_args.packing:
            model_inputs = self.tokenizer(
                text_examples,
                add_special_tokens=False,
                max_length=self.data_args.cutoff_len,
                truncation=True,
            )
        else:
            batch_input_ids, batch_labels = [], []
            lengths = []
            for text in text_examples:
                input_ids = self.tokenizer.encode(text, add_special_tokens=False)
                labels = input_ids[:]
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                lengths.append(len(input_ids))

            model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            knapsacks = packing_conversation(lengths, self.data_args.cutoff_len)
            for knapsack in knapsacks:
                packed_input_ids, packed_attention_masks, packed_labels = [], [], []
                for i, index in enumerate(knapsack):
                    packed_input_ids += batch_input_ids[index]
                    packed_labels += batch_labels[index]
                    if self.data_args.neat_packing:
                        packed_attention_masks += [i + 1] * len(
                            batch_input_ids[index]
                        )  # start from 1
                    else:
                        packed_attention_masks += [1] * len(batch_input_ids[index])

                if len(packed_input_ids) < self.data_args.cutoff_len:
                    pad_length = self.data_args.cutoff_len - len(packed_input_ids)
                    packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                    packed_labels += [IGNORE_INDEX] * pad_length
                    if self.data_args.neat_packing:
                        packed_attention_masks += [0] * pad_length
                    else:
                        packed_attention_masks += [
                            1
                        ] * pad_length  # more efficient flash_attn
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
    
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
