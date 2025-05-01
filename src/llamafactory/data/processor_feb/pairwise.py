#
# Created on Sat Apri 25 2025
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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple


from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, AudioExample, process_audio_messages


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ..template_feb import TemplateFeb
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class TextPairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        self.template: TemplateFeb

        chosen_messages = prompt + [response[0]]
        rejected_messages = prompt + [response[1]]
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * len(prompt_ids) + rejected_ids

        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
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
            "audios": []
        }
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning(
                    "Dropped invalid example: {}".format(
                        examples["_prompt"][i] + examples["_response"][i]
                    )
                )
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )

            assert len(chosen_input_ids) == len(chosen_labels)
            assert len(rejected_input_ids) == len(rejected_labels)

            if len(chosen_input_ids) < self.data_args.cutoff_len:
                pad_length = self.data_args.cutoff_len - len(chosen_input_ids)
                chosen_input_ids += [self.tokenizer.pad_token_id] * pad_length
                chosen_labels += [IGNORE_INDEX] * pad_length
            elif len(chosen_input_ids) > self.data_args.cutoff_len:
                continue

            if len(rejected_input_ids) < self.data_args.cutoff_len:
                pad_length = self.data_args.cutoff_len - len(rejected_input_ids)
                rejected_input_ids += [self.tokenizer.pad_token_id] * pad_length
                rejected_labels += [IGNORE_INDEX] * pad_length
            elif len(rejected_input_ids) > self.data_args.cutoff_len:
                continue

            assert (
                len(chosen_input_ids)
                == len(chosen_labels)
                == len(rejected_input_ids)
                == len(rejected_labels)
                == self.data_args.cutoff_len
            )

            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["videos"].append(examples["_audios"][i])

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


class AudioPairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        self.template: TemplateFeb
        chosen_messages = prompt + [response[0]]
        rejected_messages = prompt + [response[1]]

        prefix_ids = self.template.encode_system(tokenizer=self.tokenizer, system=system, tools=tools)
        chosen_audio_example: AudioExample = process_audio_messages(
            messages=chosen_messages,
            template=self.template,
            tokenizer=self.tokenizer,
            mask_history=self.data_args.mask_history
        )
        rejected_audio_example: AudioExample = process_audio_messages(
            messages=rejected_messages,
            template=self.template,
            tokenizer=self.tokenizer,
            mask_history=self.data_args.mask_history
        )

        # chosen
        chosen_input_ids = prefix_ids + chosen_audio_example.text_input_ids
        chosen_valid_tokens_pos = [pos+len(prefix_ids) for pos in chosen_audio_example.valid_tokens_pos]
        chosen_decoder_input_ids = chosen_audio_example.audio_codes_ids
        chosen_encoder_decoder_attention_mask = chosen_audio_example.t2a_attention_mask
        chosen_decoder_labels = chosen_audio_example.audio_codes_labels

        # rejected
        rejected_input_ids = prefix_ids + rejected_audio_example.text_input_ids
        rejected_valid_tokens_pos = [pos+len(prefix_ids) for pos in rejected_audio_example.valid_tokens_pos]
        rejected_decoder_input_ids = rejected_audio_example.audio_codes_ids
        rejected_encoder_decoder_attention_mask = rejected_audio_example.t2a_attention_mask
        rejected_decoder_labels = rejected_audio_example.audio_codes_labels

        return (
            chosen_input_ids, chosen_valid_tokens_pos, chosen_decoder_input_ids, chosen_encoder_decoder_attention_mask, chosen_decoder_labels,
            rejected_input_ids, rejected_valid_tokens_pos, rejected_decoder_input_ids, rejected_encoder_decoder_attention_mask, rejected_decoder_labels
        )

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = {
            "chosen_input_ids": [], "chosen_attention_mask": [], "chosen_valid_tokens_pos": [],
            "chosen_decoder_input_ids": [], "chosen_decoder_labels": [],
            "chosen_decoder_attention_mask": [], "chosen_encoder_decoder_attention_mask": [],
            "rejected_input_ids": [], "rejected_attention_mask": [], "rejected_valid_tokens_pos": [],
            "rejected_decoder_input_ids": [], "rejected_decoder_labels": [],
            "rejected_decoder_attention_mask": [], "rejected_encoder_decoder_attention_mask": [],
            "images": [], "videos": [], "audios": []
        }
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning(
                    "Dropped invalid example: {}".format(
                        examples["_prompt"][i] + examples["_response"][i]
                    )
                )
                continue

            (
                chosen_input_ids, chosen_valid_tokens_pos, chosen_decoder_input_ids, chosen_encoder_decoder_attention_mask, chosen_decoder_labels,
                rejected_input_ids, rejected_valid_tokens_pos, rejected_decoder_input_ids, rejected_encoder_decoder_attention_mask, rejected_decoder_labels
            ) = self._encode_data_example(
                    prompt=examples["_prompt"][i],
                    response=examples["_response"][i],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                )

            if len(chosen_input_ids) < self.data_args.cutoff_len:
                pad_length = self.data_args.cutoff_len - len(chosen_input_ids)
                chosen_input_ids += [self.tokenizer.pad_token_id] * pad_length
            
            if len(rejected_input_ids) < self.data_args.cutoff_len:
                pad_length = self.data_args.cutoff_len - len(rejected_input_ids)
                rejected_input_ids += [self.tokenizer.pad_token_id] * pad_length

            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_valid_tokens_pos"].append(chosen_valid_tokens_pos if chosen_valid_tokens_pos else None)
            model_inputs["chosen_decoder_input_ids"].append(chosen_decoder_input_ids if chosen_decoder_input_ids else None)
            model_inputs["chosen_decoder_attention_mask"].append([1] * len(chosen_decoder_input_ids[0]) if chosen_decoder_input_ids else None)
            model_inputs["chosen_encoder_decoder_attention_mask"].append(chosen_encoder_decoder_attention_mask if chosen_encoder_decoder_attention_mask else None)
            model_inputs["chosen_decoder_labels"].append(chosen_decoder_labels if chosen_decoder_labels else None)

            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_valid_tokens_pos"].append(rejected_valid_tokens_pos if rejected_valid_tokens_pos else None)
            model_inputs["rejected_decoder_input_ids"].append(rejected_decoder_input_ids if rejected_decoder_input_ids else None)
            model_inputs["rejected_decoder_attention_mask"].append([1] * len(rejected_decoder_input_ids[0]) if rejected_decoder_input_ids else None)
            model_inputs["rejected_encoder_decoder_attention_mask"].append(rejected_encoder_decoder_attention_mask if rejected_encoder_decoder_attention_mask else None)
            model_inputs["rejected_decoder_labels"].append(rejected_decoder_labels if rejected_decoder_labels else None)

            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["videos"].append(examples["_audios"][i])

        return model_inputs

    def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)))
