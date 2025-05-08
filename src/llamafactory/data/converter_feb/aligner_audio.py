#
# Created on Thu Feb 25 2025
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


import json
import soundfile as sf
from typing import TYPE_CHECKING, Any, Dict

from ...extras import logging
from ..data_utils import Role
from .utils import (
    process_audio_bytes,
    split_user_audio,
    resample_audio_array
)


if TYPE_CHECKING:
    from ...hparams import DataArguments
    from ..parser_feb import DatasetAttr


logger = logging.get_logger(__name__)


TARGET_SAMPLE_RATE = 24000


def convert_avater_audio(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.user_audio_tag: Role.USER_AUDIO.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.assistant_audio_tag: Role.ASSISTANT_AUDIO.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
        dataset_attr.mask_tag: Role.MASK.value,
    }
    even_tags = (dataset_attr.user_tag, dataset_attr.user_audio_tag, dataset_attr.observation_tag)
    odd_tags = (dataset_attr.assistant_tag, dataset_attr.assistant_audio_tag, dataset_attr.function_tag, dataset_attr.mask_tag)
    accept_tags = (even_tags, odd_tags)
    messages = example[dataset_attr.messages]
    if (
        dataset_attr.system_tag
        and len(messages) != 0
        and messages[0][dataset_attr.role_tag].lower() == dataset_attr.system_tag
    ):
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = dataset_attr.system if dataset_attr.system else ""

    aligned_messages = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag].lower() not in accept_tags[turn_idx % 2]:
            print(f"Invalid role tag in {messages}.")
            broken_data = True

        if message[dataset_attr.role_tag].lower() == dataset_attr.user_audio_tag:
            aligned_message = {
                "role": tag_mapping[message[dataset_attr.role_tag].lower()],
                "content": []
            }
            for item in json.loads(message[dataset_attr.content_tag]):
                if item["type"] != "audio":
                    aligned_message["content"].append(item)
                else:
                    try:
                        audio_array, sample_rate = sf.read(item["file"])
                    except:
                        broken_data = True
                        break
                    aligned_message["content"] += split_user_audio(
                        audio_array,
                        sample_rate,
                        target_sr=16000,
                        duration=30
                    )
            aligned_messages.append(aligned_message)
        elif message[dataset_attr.role_tag].lower() == dataset_attr.assistant_audio_tag:
            audios = []
            for item in message["audios"]:
                try:
                    array, sample_rate = sf.read(item["file"])
                except:
                    broken_data = True
                    break
                audios.append({
                    "id": item["id"],
                    "array": resample_audio_array(
                        array,
                        sample_rate,
                        target_sr=TARGET_SAMPLE_RATE
                    ),
                    "split": item["split"]
                })
            aligned_messages.append(
                {
                    "role": tag_mapping[message[dataset_attr.role_tag].lower()],
                    "content": message[dataset_attr.content_tag].strip(),
                    "audios": audios
                }
            )
        else:
            aligned_messages.append(
                {
                    "role": tag_mapping[message[dataset_attr.role_tag].lower()],
                    "content": message[dataset_attr.content_tag],
                }
            )

        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        print("Skipping this abnormal example.")
        prompt, response = [], []

    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": None,
        "_videos": None,
    }
    return output
