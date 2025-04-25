#
# Created on Thu Apri 25 2025
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

import bisect
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Optional

from ..data_utils import Role
from ...extras import logging
from ...extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template
    from ..template_feb import TemplateFeb


logger = logging.get_logger(__name__)


@dataclass
class DatasetProcessor(ABC):
    r"""A class for data processors."""

    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build model inputs from the examples."""
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""Print a data example to stdout."""
        ...


def search_for_fit(numbers: list[int], capacity: int) -> int:
    r"""Find the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem."""
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""Compute the real sequence length after truncation by the cutoff_len."""
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


"""
Feb Utils
"""
@dataclass
class AudioExampleBase:
    # text input/output
    text_input_ids: list[int]
    text_labels: list[int]

    # audio input
    audio_features: list[dict]
    audio_positions: list[list[int]]

    # audio output
    audio_codes_ids: list[list[int]]
    audio_codes_labels: list[int]
    valid_tokens_pos: list[list[int]]
    t2a_attention_mask: list[list[int]]


class AudioExample(AudioExampleBase, dict):
    def __post_init__(self):
        # populate the dict
        for f in fields(self):
            self[f.name] = getattr(self, f.name)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AudioExample' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if key in {f.name for f in fields(self)}:
            super().__setattr__(key, value)
            self[key] = value
        else:
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if key in {f.name for f in fields(self)}:
            super().__setattr__(key, value)


def packing_conversation(numbers: list[int], capacity: int) -> list[list[int]]:
    r"""An efficient packing algorithm with binary search for the conversation dataset."""
    knapsacks = []
    data_idx_index = 0
    while data_idx_index < len(numbers):
        current_knapsack = []
        remaining_capacity = capacity
        while remaining_capacity > 0 and data_idx_index < len(numbers):
            data_length = numbers[data_idx_index]
            if data_length >= capacity:
                if remaining_capacity == capacity:
                    current_knapsack.append(data_idx_index)
                    data_idx_index += 1
                    remaining_capacity -= data_length
                else:
                    current_knapsack.append(data_idx_index)
                    remaining_capacity -= data_length
            elif data_length > remaining_capacity:
                current_knapsack.append(data_idx_index)
                remaining_capacity -= data_length
            else:
                current_knapsack.append(data_idx_index)
                data_idx_index += 1
                remaining_capacity -= data_length

        knapsacks.append(current_knapsack)

    return knapsacks


def process_audio_messages(
    messages: list[dict[str]],
    template: TemplateFeb,
    tokenizer: "PreTrainedTokenizer",
    mask_history: bool
) -> AudioExample:
    text_input_ids, text_labels  = [], []
    audio_features, audio_positions = [], []
    valid_tokens_pos, audio_codes_ids, audio_codes_labels = [], [], []
    t2a_attention_mask = []

    retry_time = 0
    while retry_time < 10:
        try:
            encoded_pairs = template.encode_avater_audio(tokenizer=tokenizer, messages=messages)
            break
        except Exception as e:
            retry_time += 1
            if retry_time >= 10:
                logger.warning_rank0(e)
                return None

    audio_start_pos = 0
    text_pairs = [(messages[i], messages[i + 1]) for i in range(0, len(messages), 2)]
    for turn_idx, (source_dict, target_dict) in enumerate(encoded_pairs):
        # text
        source_token_ids = source_dict["token_ids"]
        target_token_ids = target_dict["token_ids"]

        source_text_len = len(source_token_ids)
        target_text_len = len(target_token_ids)

        source_text_label = [IGNORE_INDEX] * source_text_len

        if mask_history and turn_idx != len(encoded_pairs) - 1:
            target_text_label = [IGNORE_INDEX] * target_text_len
        elif text_pairs[turn_idx][1]["role"] == Role.MASK.value:
            target_text_label = [IGNORE_INDEX] * target_text_len
        else:
            target_text_label = target_token_ids[:]

        # audio input
        if "audio_features" in source_dict:
            audio_features += source_dict["audio_features"]
            audio_positions += [[audio_start_pos+audio_start, audio_length] for audio_start, audio_length in source_dict["audio_positions"]]

        # audio output
        if turn_idx == len(encoded_pairs) - 1 and "audio_codes" in target_dict:
            if template.name == "llama3":
                start_prefix_idx = 3
            else:
                start_prefix_idx = 0

            try:
                t2a_attention_mask = tokenizer.convert_t2a_attention_mask(len(target_token_ids[start_prefix_idx:]), len(target_dict["audio_codes"][0]))
            except Exception as e:
                logger.warning_rank0(e)
                return None

            valid_tokens_pos = [idx for idx in range(
                len(text_labels)+source_text_len+start_prefix_idx, len(text_labels)+source_text_len+len(target_token_ids)
            )]
            audio_codes_ids = target_dict["audio_codes"]
            audio_codes_labels = copy.deepcopy(target_dict["audio_codes"])

        text_input_ids += source_token_ids + target_token_ids
        text_labels += source_text_label + target_text_label
        audio_start_pos += len(text_input_ids)
    
    assert len(text_input_ids) == len(text_labels), "The length of text_input_ids should equal with labels' length!"
    assert len(audio_codes_ids) == len(audio_codes_labels), "The length of audio_codes_ids should equal with labels' length!"

    right = len(audio_codes_ids)==0 or (
        len(t2a_attention_mask) == len(audio_codes_ids[0])
        and len(t2a_attention_mask[0]) == len(valid_tokens_pos)
        and len(audio_codes_ids) == len(audio_codes_labels)
        and len(audio_codes_ids[0]) == len(audio_codes_labels[0])
    )
    if not right:
        return None

    return AudioExample(
        text_input_ids=text_input_ids, text_labels=text_labels,
        audio_features=audio_features, audio_positions=audio_positions,
        audio_codes_ids=audio_codes_ids, audio_codes_labels=audio_codes_labels,
        valid_tokens_pos=valid_tokens_pos, t2a_attention_mask=t2a_attention_mask
    )
