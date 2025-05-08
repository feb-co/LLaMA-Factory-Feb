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

import random
import copy
from typing import TYPE_CHECKING, Any, Dict, List

from ...extras import logging
from ..data_utils import Role
from .utils import (
    process_audio_bytes,
    resample_audio_array
)
from .prompts import SYSTEM_SYTLE_PROMPT, SYSTEM_TTS_PROMPT, USER_TTS_PROMPT


if TYPE_CHECKING:
    from ...hparams import DataArguments
    from ..parser_feb import DatasetAttr


TARGET_SAMPLE_RATE=24000
AUDIO_SPLIT_TAG = [
    "",
    "<|SHORT_WAIT|>",
    "<|LONG_WAIT|>"
]


logger = logging.get_logger(__name__)


def convert_avater_audio_arrow_tts(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    if "packed_LargeScaleASR" in dataset_attr.dataset_key:
        return convert_packed_LargeScaleASR(example, dataset_attr, data_args)
    elif "packed_parler" in dataset_attr.dataset_key:
        return convert_packed_parler_tts(example, dataset_attr, data_args)
    elif "packed_text" in dataset_attr.dataset_key:
        return convert_tts_packed_text(example, dataset_attr, data_args)
    elif "packed_segment" in dataset_attr.dataset_key:
        return convert_tts_packed_segment(example, dataset_attr, data_args)
    elif "packed_dialogue" in dataset_attr.dataset_key:
        return convert_tts_packed_dialogue(example, dataset_attr, data_args)

    if "LargeScaleASR" in dataset_attr.dataset_key:
        return convert_LargeScaleASR(example, dataset_attr, data_args)
    elif "parler_tts" in dataset_attr.dataset_key:
        return convert_parler_tts(example, dataset_attr, data_args)
    elif "text_" in dataset_attr.dataset_key:
        return convert_tts_text(example, dataset_attr, data_args)
    elif "segment_" in dataset_attr.dataset_key:
        return convert_tts_segment(example, dataset_attr, data_args)
    elif "dialogue_" in dataset_attr.dataset_key:
        return convert_tts_dialogue(example, dataset_attr, data_args)
    else:
        raise NotImplementedError


def convert_LargeScaleASR(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    system_list = dataset_attr.system_list if dataset_attr.system_list else ["You are a helpful assistant."]
    system = random.choice(system_list)
    example_id = example["ID"]
    if "sex" in example and example["sex"]:
        style = random.choice(SYSTEM_SYTLE_PROMPT).format(style=example["sex"].replace("_", " "))
    else:
        style = ""

    audio_text = random.choice([example["text"], example["text"].lower()])
    audio_text = audio_text.strip()

    flag_list = [0, 1]
    flag = random.choice(flag_list)
    if flag == 0:
        system += (random.choice(["\n", "\n\n", " "]) + style + random.choice(["\n", "\n\n", " "]) + random.choice(SYSTEM_TTS_PROMPT))
        user_prompt = audio_text
    else:
        system += (random.choice(["\n", "\n\n", " "]) + style)
        user_prompt = random.choice(USER_TTS_PROMPT).format(text=audio_text)

    system = system.strip().strip("\n")

    try:
        audio_array, sample_rate = process_audio_bytes(example["wav"]["bytes"])
        prompt = [{
            "role": Role.USER.value,
            "content": user_prompt,
        }]
        response = [{
            "role": Role.ASSISTANT_AUDIO.value,
            "content": audio_text,
            "audios": [{
                "id": example_id,
                "array": resample_audio_array(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=TARGET_SAMPLE_RATE
                ),
                "split": ""
            }]
        }]
    except Exception as e:
        logger.warning_rank0(e)
        prompt = []
        response = []

    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": None,
        "_videos": None,
    }
    return output


def convert_packed_LargeScaleASR(
    examples: Dict[str, List[Any]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    cut_off_len = random.choice([int(length) for length in range(data_args.cutoff_len//4, data_args.cutoff_len, 256)])
    flag_list = [0, 1]
    flag = random.choice(flag_list)

    outputs = {
        "_prompt": [],
        "_response": [],
        "_system": [],
        "_tools": [],
        "_images": [],
        "_videos": [],
    }

    prompt = []
    current_seq_length = 0
    for i in range(len(examples["ID"])):
        system_list = dataset_attr.system_list if dataset_attr.system_list else ["You are a helpful assistant."]
        system = random.choice(system_list)
        example_id = examples["ID"][i]
        if "sex" in examples and examples["sex"][i]:
            style = random.choice(SYSTEM_SYTLE_PROMPT).format(style=examples["sex"][i].replace("_", " "))
        else:
            style = ""

        audio_text = random.choice([examples["text"][i], examples["text"][i].lower()])
        audio_text = audio_text.strip()

        if flag == 0:
            system += (random.choice(["\n", "\n\n", " "]) + style + random.choice(["\n", "\n\n", " "]) + random.choice(SYSTEM_TTS_PROMPT))
            user_prompt = audio_text
        else:
            system += (random.choice(["\n", "\n\n", " "]) + style)
            user_prompt = random.choice(USER_TTS_PROMPT).format(text=audio_text)

        system = system.strip().strip("\n").replace("  ", " ").replace("\n\n\n", "\n\n")

        if (current_seq_length + len(system.split()) + len(user_prompt.split()) + len(audio_text.split()) + 8 >= cut_off_len) or (i == len(examples["ID"])-1):
            try:
                audio_array, sample_rate = process_audio_bytes(examples["wav"][i]["bytes"])
                prompt += [{
                    "role": Role.USER.value,
                    "content": user_prompt,
                }]
                response = [{
                    "role": Role.ASSISTANT_AUDIO.value,
                    "content": audio_text,
                    "audios": [{
                        "id": example_id,
                        "array": resample_audio_array(
                            audio_array,
                            orig_sr=sample_rate,
                            target_sr=TARGET_SAMPLE_RATE
                        ),
                        "split": ""
                    }]
                }]
                outputs["_prompt"].append(prompt)
                outputs["_response"].append(response)
                outputs["_system"].append(system)
                outputs["_tools"].append("")
                outputs["_images"].append(None)
                outputs["_videos"].append(None)
                prompt = []
                current_seq_length = 0
            except Exception as e:
                logger.warning_rank0(e)
                continue
        else:
            current_seq_length += len(user_prompt.split()) + len(audio_text.split()) + 8
            prompt += [
                {
                    "role": Role.USER.value,
                    "content": user_prompt,
                },
                {
                    "role": Role.ASSISTANT.value,
                    "content": audio_text,
                }
            ]

    return outputs


def convert_parler_tts(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    system_list = dataset_attr.system_list if dataset_attr.system_list else ["You are a helpful assistant."]
    system = random.choice(system_list)
    example_id = example["speaker_id"]
    if "audio_description" in example and example["audio_description"]:
        style = random.choice(SYSTEM_SYTLE_PROMPT).format(style=example["audio_description"].replace("_", " "))
    else:
        style = ""

    audio_text = random.choice([example["transcript"], example["transcript"].lower()])
    audio_text = audio_text.strip()

    flag_list = [0, 1]
    flag = random.choice(flag_list)
    if flag == 0:
        system += (random.choice(["\n", "\n\n", " "]) + style + random.choice(["\n", "\n\n", " "]) + random.choice(SYSTEM_TTS_PROMPT))
        user_prompt = audio_text
    else:
        system += (random.choice(["\n", "\n\n", " "]) + style)
        user_prompt = random.choice(USER_TTS_PROMPT).format(text=audio_text)

    system = system.strip().strip("\n").replace("  ", " ").replace("\n\n\n", "\n\n")

    try:
        audio_array = example["audio"]["array"]
        sample_rate = example["audio"]["sampling_rate"]
        prompt = [{
            "role": Role.USER.value,
            "content": user_prompt,
        }]
        response = [{
            "role": Role.ASSISTANT_AUDIO.value,
            "content": audio_text,
            "audios": [{
                "id": example_id,
                "array": resample_audio_array(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=TARGET_SAMPLE_RATE
                ),
                "split": ""
            }]
        }]
    except Exception as e:
        logger.warning_rank0(e)
        prompt = []
        response = []

    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": None,
        "_videos": None,
    }
    return output


def convert_packed_parler_tts(
    examples: Dict[str, List[Any]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    cut_off_len = random.choice([int(length) for length in range(data_args.cutoff_len//4, data_args.cutoff_len, 256)])
    flag_list = [0, 1]
    flag = random.choice(flag_list)

    outputs = {
        "_prompt": [],
        "_response": [],
        "_system": [],
        "_tools": [],
        "_images": [],
        "_videos": [],
    }
    
    prompt = []
    current_seq_length = 0
    for i in range(len(examples["speaker_id"])):
        system_list = dataset_attr.system_list if dataset_attr.system_list else ["You are a helpful assistant."]
        system = random.choice(system_list)
        example_id = examples["speaker_id"][i]
        if "audio_description" in examples and examples["audio_description"][i]:
            style = random.choice(SYSTEM_SYTLE_PROMPT).format(style=examples["audio_description"][i].replace("_", " "))
        else:
            style = ""

        audio_text = random.choice([examples["transcript"][i], examples["transcript"][i].lower()])
        audio_text = audio_text.strip()

        if flag == 0:
            system += (random.choice(["\n", "\n\n", " "]) + style + random.choice(["\n", "\n\n", " "]) + random.choice(SYSTEM_TTS_PROMPT))
            user_prompt = audio_text
        else:
            system += (random.choice(["\n", "\n\n", " "]) + style)
            user_prompt = random.choice(USER_TTS_PROMPT).format(text=audio_text)

        system = system.strip().strip("\n").replace("  ", " ").replace("\n\n\n", "\n\n")

        if (current_seq_length + len(system.split()) + len(user_prompt.split()) + len(audio_text.split()) + 8 >= cut_off_len) or (i == len(examples["speaker_id"])-1):
            try:
                audio_array = examples["audio"][i]["array"]
                sample_rate = examples["audio"][i]["sampling_rate"]
                prompt += [{
                    "role": Role.USER.value,
                    "content": user_prompt,
                }]
                response = [{
                    "role": Role.ASSISTANT_AUDIO.value,
                    "content": audio_text,
                    "audios": [{
                        "id": example_id,
                        "array": resample_audio_array(
                            audio_array,
                            orig_sr=sample_rate,
                            target_sr=TARGET_SAMPLE_RATE
                        ),
                        "split": ""
                    }]
                }]
                outputs["_prompt"].append(prompt)
                outputs["_response"].append(response)
                outputs["_system"].append(system)
                outputs["_tools"].append("")
                outputs["_images"].append(None)
                outputs["_videos"].append(None)
                prompt = []
                current_seq_length = 0
            except Exception as e:
                logger.warning_rank0(e)
                prompt = []
                response = []
        else:
            current_seq_length += len(user_prompt.split()) + len(audio_text.split()) + 8
            prompt += [
                {
                    "role": Role.USER.value,
                    "content": user_prompt,
                },
                {
                    "role": Role.ASSISTANT.value,
                    "content": audio_text,
                }
            ]

    return outputs


def convert_tts_text(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    system_list = dataset_attr.system_list if dataset_attr.system_list else ["You are a helpful assistant."]
    system = random.choice(system_list)
    try:
        example_id = example["id"]
    except:
        example_id = "0"

    audio_text = random.choice([example["text"], example["text"].lower()])
    audio_text = audio_text.strip()

    flag_list = [0, 1]
    flag = random.choice(flag_list)
    if flag == 0:
        system += (random.choice(["\n", "\n\n", " "]) + random.choice(SYSTEM_TTS_PROMPT))
        user_prompt = audio_text
    else:
        user_prompt = random.choice(USER_TTS_PROMPT).format(text=audio_text)

    system = system.strip().strip("\n")

    try:
        audio_array = example["audio"]["array"]
        sample_rate = example["audio"]["sample_rate"] if "sample_rate" in example["audio"] else example["audio"]["sampling_rate"]
        prompt = [{
            "role": Role.USER.value,
            "content": user_prompt,
        }]
        response = [{
            "role": Role.ASSISTANT_AUDIO.value,
            "content": audio_text,
            "audios": [{
                "id": example_id,
                "array": resample_audio_array(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=TARGET_SAMPLE_RATE
                ),
                "split": ""
            }]
        }]
    except Exception as e:
        logger.warning_rank0(e)
        prompt = []
        response = []

    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": None,
        "_videos": None,
    }
    return output


def convert_tts_packed_text(
    examples: Dict[str, List[Any]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    cut_off_len = random.choice([int(length) for length in range(data_args.cutoff_len//4, data_args.cutoff_len, 256)])
    flag_list = [0, 1]
    flag = random.choice(flag_list)

    outputs = {
        "_prompt": [],
        "_response": [],
        "_system": [],
        "_tools": [],
        "_images": [],
        "_videos": [],
    }

    prompt = []
    current_seq_length = 0
    for i in range(len(examples["id"])):
        system_list = dataset_attr.system_list if dataset_attr.system_list else ["You are a helpful assistant."]
        system = random.choice(system_list)
        example_id = examples["id"][i]

        audio_text = random.choice([examples["text"][i], examples["text"][i].lower()])
        audio_text = audio_text.strip()

        if flag == 0:
            system += (random.choice(["\n", "\n\n", " "]) + random.choice(SYSTEM_TTS_PROMPT))
            user_prompt = audio_text
        else:
            user_prompt = random.choice(USER_TTS_PROMPT).format(text=audio_text)

        system = system.strip().strip("\n")

        if (current_seq_length + len(system.split()) + len(user_prompt.split()) + len(audio_text.split()) + 8 >= cut_off_len) or (i == len(examples["id"])-1):
            try:
                audio_array = examples["audio"][i]["array"]
                sample_rate = examples["audio"][i]["sample_rate"] if "sample_rate" in examples["audio"][i] else examples["audio"][i]["sampling_rate"]
                prompt += [{
                    "role": Role.USER.value,
                    "content": user_prompt,
                }]
                response = [{
                    "role": Role.ASSISTANT_AUDIO.value,
                    "content": audio_text,
                    "audios": [{
                        "id": example_id,
                        "array": resample_audio_array(
                            audio_array,
                            orig_sr=sample_rate,
                            target_sr=TARGET_SAMPLE_RATE
                        ),
                        "split": ""
                    }]
                }]
                outputs["_prompt"].append(prompt)
                outputs["_response"].append(response)
                outputs["_system"].append(system)
                outputs["_tools"].append("")
                outputs["_images"].append(None)
                outputs["_videos"].append(None)
                prompt = []
                current_seq_length = 0
            except Exception as e:
                logger.warning_rank0(e)
                continue
        else:
            current_seq_length += len(user_prompt.split()) + len(audio_text.split()) + 8
            prompt += [
                {
                    "role": Role.USER.value,
                    "content": user_prompt,
                },
                {
                    "role": Role.ASSISTANT.value,
                    "content": audio_text,
                }
            ]

    return outputs


def convert_tts_segment(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    system_list = dataset_attr.system_list if dataset_attr.system_list else ["You are a helpful assistant."]
    system = random.choice(system_list)

    audio_text = ""
    audios = []
    for idx, wav_text in enumerate(example["segments"]):
        wav_id = example["sentence_ids"][idx]
        if idx == 0:
            audio_text += wav_text
            audios.append(
                {
                    "id": wav_id,
                    "array": resample_audio_array(
                        example["sentence_audios"][idx]["array"],
                        orig_sr=example["sentence_audios"][idx]["sample_rate"],
                        target_sr=TARGET_SAMPLE_RATE
                    ),
                    "split": AUDIO_SPLIT_TAG[0]
                }
            )
        else:
            split_tag = random.choice(AUDIO_SPLIT_TAG)
            audio_text += f" {split_tag} ".replace("  ", " ") + wav_text
            audios.append(
                {
                    "id": wav_id,
                    "array": resample_audio_array(
                        example["sentence_audios"][idx]["array"],
                        orig_sr=example["sentence_audios"][idx]["sample_rate"],
                        target_sr=TARGET_SAMPLE_RATE
                    ),
                    "split": split_tag
                }
            )

    audio_text = random.choice([audio_text, audio_text.lower()])
    audio_text = audio_text.strip()
    flag_list = [0, 1]
    flag = random.choice(flag_list)
    if flag == 0:
        system += (random.choice(["\n", "\n\n", " "]) + random.choice(SYSTEM_TTS_PROMPT))
        user_prompt = audio_text
    else:
        user_prompt = random.choice(USER_TTS_PROMPT).format(text=audio_text)

    system = system.strip().strip("\n")

    prompt = [{
        "role": Role.USER.value,
        "content": user_prompt,
    }]
    response = [{
        "role": Role.ASSISTANT_AUDIO.value,
        "content": audio_text,
        "audios": audios
    }]

    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": None,
        "_videos": None,
    }
    return output


def convert_tts_packed_segment(
    examples: Dict[str, List[Any]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    cut_off_len = random.choice([int(length) for length in range(data_args.cutoff_len//4, data_args.cutoff_len, 256)])
    flag_list = [0, 1]
    flag = random.choice(flag_list)

    outputs = {
        "_prompt": [],
        "_response": [],
        "_system": [],
        "_tools": [],
        "_images": [],
        "_videos": [],
    }

    prompt = []
    current_seq_length = 0
    for i in range(len(examples["segments"])):
        system_list = dataset_attr.system_list if dataset_attr.system_list else ["You are a helpful assistant."]
        system = random.choice(system_list)

        audio_text = ""
        audios = []
        for idx, wav_text in enumerate(examples["segments"][i]):
            wav_id = examples["sentence_ids"][i][idx]
            if idx == 0:
                audio_text += wav_text
                audios.append(
                    {
                        "id": wav_id,
                        "array": resample_audio_array(
                            examples["sentence_audios"][i][idx]["array"],
                            orig_sr=examples["sentence_audios"][i][idx]["sample_rate"],
                            target_sr=TARGET_SAMPLE_RATE
                        ),
                        "split": AUDIO_SPLIT_TAG[0]
                    }
                )
            else:
                split_tag = random.choice(AUDIO_SPLIT_TAG)
                audio_text += f" {split_tag} ".replace("  ", " ") + wav_text
                audios.append(
                    {
                        "id": wav_id,
                        "array": resample_audio_array(
                            examples["sentence_audios"][i][idx]["array"],
                            orig_sr=examples["sentence_audios"][i][idx]["sample_rate"],
                            target_sr=TARGET_SAMPLE_RATE
                        ),
                        "split": split_tag
                    }
                )

        audio_text = random.choice([audio_text, audio_text.lower()])
        audio_text = audio_text.strip()
        if flag == 0:
            system += (random.choice(["\n", "\n\n", " "]) + random.choice(SYSTEM_TTS_PROMPT))
            user_prompt = audio_text
        else:
            user_prompt = random.choice(USER_TTS_PROMPT).format(text=audio_text)

        system = system.strip().strip("\n")
        
        if (current_seq_length + len(system.split()) + len(user_prompt.split()) + len(audio_text.split()) + 8 >= cut_off_len) or (i == len(examples["segments"])-1):
            prompt += [{
                "role": Role.USER.value,
                "content": user_prompt,
            }]
            response = [{
                "role": Role.ASSISTANT_AUDIO.value,
                "content": audio_text,
                "audios": audios
            }]
            outputs["_prompt"].append(prompt)
            outputs["_response"].append(response)
            outputs["_system"].append(system)
            outputs["_tools"].append("")
            outputs["_images"].append(None)
            outputs["_videos"].append(None)
            prompt = []
            current_seq_length = 0
        else:
            current_seq_length += len(user_prompt.split()) + len(audio_text.split()) + 8
            prompt += [
                {
                    "role": Role.USER.value,
                    "content": user_prompt,
                },
                {
                    "role": Role.ASSISTANT.value,
                    "content": audio_text,
                }
            ]

    return outputs


def convert_tts_dialogue(
    examples: Dict[str, List[Any]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.user_audio_tag: Role.USER_AUDIO.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.assistant_audio_tag: Role.ASSISTANT_AUDIO.value,
    }

    outputs = {
        "_prompt": [],
        "_response": [],
        "_system": [],
        "_tools": [],
        "_images": [],
        "_videos": [],
    }
    for i in range(len(examples["id"])):
        system_list = dataset_attr.system_list if dataset_attr.system_list else ["You are a helpful assistant."]
        system = random.choice(system_list)

        conversations = examples["conversations"][i]
        messages = []
        for message in conversations:
            if message[dataset_attr.role_tag].lower() in [dataset_attr.user_audio_tag, dataset_attr.user_tag, dataset_attr.assistant_tag]:
                messages.append(
                    {
                        "role": tag_mapping[message[dataset_attr.role_tag].lower()],
                        "content": message[dataset_attr.content_tag].strip(),
                    }
                )
            elif message[dataset_attr.role_tag].lower() == dataset_attr.assistant_audio_tag:
                audios = [
                    {
                        "id": audio["id"],
                        "array": resample_audio_array(
                            audio["array"],
                            audio["sample_rate"],
                            target_sr=TARGET_SAMPLE_RATE
                        ),
                        "split": audio["split"],
                    }
                    for audio in message["audios"]
                ]
                outputs["_prompt"].append(copy.deepcopy(messages))
                outputs["_response"].append([{
                    "role": Role.ASSISTANT_AUDIO.value,
                    "content": message[dataset_attr.content_tag].strip(),
                    "audios": audios
                }])
                outputs["_system"].append(system)
                outputs["_tools"].append("")
                outputs["_images"].append(None)
                outputs["_videos"].append(None)

                messages.append(
                    {
                        "role": Role.ASSISTANT.value,
                        "content": message[dataset_attr.content_tag].strip(),
                    }
                )
            else:
                raise NotImplementedError

    return outputs


def convert_tts_packed_dialogue(
    examples: Dict[str, List[Any]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    cut_off_len = random.choice([int(length) for length in range(data_args.cutoff_len//4, data_args.cutoff_len, 256)])

    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.user_audio_tag: Role.USER_AUDIO.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.assistant_audio_tag: Role.ASSISTANT_AUDIO.value,
    }

    outputs = {
        "_prompt": [],
        "_response": [],
        "_system": [],
        "_tools": [],
        "_images": [],
        "_videos": [],
    }
    messages = []
    current_seq_length = 0
    for i in range(len(examples["id"])):
        system_list = dataset_attr.system_list if dataset_attr.system_list else ["You are a helpful assistant."]
        system = random.choice(system_list)

        conversations = examples["conversations"][i]
        for message in conversations:
            if message[dataset_attr.role_tag].lower() in [dataset_attr.user_audio_tag, dataset_attr.user_tag, dataset_attr.assistant_tag]:
                messages.append(
                    {
                        "role": tag_mapping[message[dataset_attr.role_tag].lower()],
                        "content": message[dataset_attr.content_tag].strip(),
                    }
                )
                current_seq_length += len(message[dataset_attr.content_tag].split())
            elif message[dataset_attr.role_tag].lower() == dataset_attr.assistant_audio_tag and ((current_seq_length + len(message[dataset_attr.content_tag].split()) + 8 >= cut_off_len) or (i == len(examples["id"])-1)):
                audios = [
                    {
                        "id": audio["id"],
                        "array": resample_audio_array(
                            audio["array"],
                            audio["sample_rate"],
                            target_sr=TARGET_SAMPLE_RATE
                        ),
                        "split": audio["split"],
                    }
                    for audio in message["audios"]
                ]

                outputs["_prompt"].append(messages)
                outputs["_response"].append([{
                    "role": Role.ASSISTANT_AUDIO.value,
                    "content": message[dataset_attr.content_tag].strip(),
                    "audios": audios
                }])
                outputs["_system"].append(system)
                outputs["_tools"].append("")
                outputs["_images"].append(None)
                outputs["_videos"].append(None)

                messages = []
                current_seq_length = 0
                break
            elif message[dataset_attr.role_tag].lower() == dataset_attr.assistant_audio_tag:
                messages.append(
                    {
                        "role": Role.ASSISTANT.value,
                        "content": message[dataset_attr.content_tag].strip(),
                    }
                )
                current_seq_length += len(message[dataset_attr.content_tag].split())
            else:
                raise NotImplementedError

    return outputs
