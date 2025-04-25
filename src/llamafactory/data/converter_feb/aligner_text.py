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


import os
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from ...extras import logging
from ..data_utils import Role


if TYPE_CHECKING:
    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..parser_feb import DatasetAttr


logger = logging.get_logger(__name__)


def _convert_images(
    images: Union["ImageInput", Sequence["ImageInput"]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Optional[List["ImageInput"]]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    if not isinstance(images, list):
        images = [images]
    elif len(images) == 0:
        return None
    else:
        images = images[:]

    if dataset_attr.load_from in ["script", "file"]:
        for i in range(len(images)):
            if isinstance(images[i], str) and os.path.isfile(os.path.join(data_args.image_dir, images[i])):
                images[i] = os.path.join(data_args.image_dir, images[i])

    return images


def _convert_videos(
    videos: Union["VideoInput", Sequence["VideoInput"]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Optional[List["VideoInput"]]:
    r"""
    Optionally concatenates video path to dataset dir when loading from local disk.
    """
    if not isinstance(videos, list):
        videos = [videos]
    elif len(videos) == 0:
        return None
    else:
        videos = videos[:]

    if dataset_attr.load_from in ["script", "file"]:
        for i in range(len(videos)):
            if isinstance(videos[i], str) and os.path.isfile(os.path.join(data_args.image_dir, videos[i])):
                videos[i] = os.path.join(data_args.image_dir, videos[i])

    return videos


def convert_alpaca(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts alpaca format dataset to the standard format.
    """
    prompt = []
    if dataset_attr.history and isinstance(example[dataset_attr.history], list):
        for old_prompt, old_response in example[dataset_attr.history]:
            prompt.append({"role": Role.USER.value, "content": old_prompt})
            prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

    query = []
    if dataset_attr.prompt and example[dataset_attr.prompt]:
        query.append(example[dataset_attr.prompt])

    if dataset_attr.query and example[dataset_attr.query]:
        query.append(example[dataset_attr.query])

    prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag], bool):  # kto example
        response = [{"role": Role.ASSISTANT.value, "content": example[dataset_attr.response]}]
        if example[dataset_attr.kto_tag]:
            response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        else:
            response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
    elif (
        dataset_attr.ranking
        and isinstance(example[dataset_attr.chosen], str)
        and isinstance(example[dataset_attr.rejected], str)
    ):  # pairwise example
        response = [
            {"role": Role.ASSISTANT.value, "content": example[dataset_attr.chosen]},
            {"role": Role.ASSISTANT.value, "content": example[dataset_attr.rejected]},
        ]
    elif dataset_attr.response and isinstance(example[dataset_attr.response], str):  # normal example
        response = [{"role": Role.ASSISTANT.value, "content": example[dataset_attr.response]}]
    else:  # unsupervised
        response = []

    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    convert_videos = partial(_convert_videos, dataset_attr=dataset_attr, data_args=data_args)
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": dataset_attr.system if dataset_attr.system else "",
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output


def convert_sharegpt(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
        dataset_attr.mask_tag: Role.MASK.value,
    }
    even_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    odd_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag, dataset_attr.mask_tag)
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
            logger.warning_rank0(f"Invalid role tag in {messages}.")
            broken_data = True

        aligned_messages.append(
            {"role": tag_mapping[message[dataset_attr.role_tag].lower()], "content": message[dataset_attr.content_tag]}
        )

    if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
        dataset_attr.ranking and len(aligned_messages) % 2 == 0
    ):
        logger.warning_rank0(f"Invalid message count in {messages}.")
        broken_data = True

    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag], bool):  # kto example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]
        if example[dataset_attr.kto_tag]:
            response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        else:
            response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
    elif (
        dataset_attr.ranking
        and isinstance(example[dataset_attr.chosen], dict)
        and isinstance(example[dataset_attr.rejected], dict)
    ):  # pairwise example
        chosen = example[dataset_attr.chosen]
        rejected = example[dataset_attr.rejected]
        if (
            chosen[dataset_attr.role_tag].lower() not in accept_tags[-1]
            or rejected[dataset_attr.role_tag].lower() not in accept_tags[-1]
        ):
            logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
            broken_data = True

        prompt = aligned_messages
        response = [
            {"role": tag_mapping[chosen[dataset_attr.role_tag].lower()], "content": chosen[dataset_attr.content_tag]},
            {"role": tag_mapping[rejected[dataset_attr.role_tag].lower()], "content": rejected[dataset_attr.content_tag]},
        ]
    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        logger.warning_rank0("Skipping this abnormal example.")
        prompt, response = [], []

    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    convert_videos = partial(_convert_videos, dataset_attr=dataset_attr, data_args=data_args)
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output


def convert_longthought(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
        dataset_attr.mask_tag: Role.MASK.value,
        dataset_attr.think_tag: Role.THINK.value,
        
    }
    first_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    second_tags = (dataset_attr.think_tag,)
    thrid_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag, dataset_attr.mask_tag)
    accept_tags = (first_tags, second_tags, thrid_tags)
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
        if message[dataset_attr.role_tag].lower() not in accept_tags[turn_idx % 3]:
            logger.warning_rank0(f"Invalid role tag in {messages}.")
            broken_data = True

        aligned_messages.append(
            {"role": tag_mapping[message[dataset_attr.role_tag].lower()], "content": message[dataset_attr.content_tag]}
        )

    if (not dataset_attr.ranking and len(aligned_messages) % 3 != 0) or (
        dataset_attr.ranking and len(aligned_messages) % 3 == 0
    ):
        logger.warning_rank0(f"Invalid message count in {messages}.")
        broken_data = True

    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag], bool):  # kto example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]
        if example[dataset_attr.kto_tag]:
            response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        else:
            response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
    elif (
        dataset_attr.ranking
        and isinstance(example[dataset_attr.chosen], dict)
        and isinstance(example[dataset_attr.rejected], dict)
    ):  # pairwise example
        chosen = example[dataset_attr.chosen]
        rejected = example[dataset_attr.rejected]
        if (
            chosen[dataset_attr.role_tag].lower() not in accept_tags[-1]
            or rejected[dataset_attr.role_tag].lower() not in accept_tags[-1]
        ):
            logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
            broken_data = True

        prompt = aligned_messages
        response = [
            {"role": tag_mapping[chosen[dataset_attr.role_tag].lower()], "content": chosen[dataset_attr.content_tag]},
            {"role": tag_mapping[rejected[dataset_attr.role_tag].lower()], "content": rejected[dataset_attr.content_tag]},
        ]
    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        logger.warning_rank0("Skipping this abnormal example.")
        prompt, response = [], []

    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    convert_videos = partial(_convert_videos, dataset_attr=dataset_attr, data_args=data_args)
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output


def convert_document(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments"
) -> Dict[str, Any]:
    r"""
    Converts document pretrain format dataset to the standard format.
    """
    prompt = []
    content = [example[dataset_attr.prefix]]
    for sentence in example[dataset_attr.document]:
        sentence = sentence.replace('\\n', '\n').rstrip()
        content.append(sentence)
    prompt.append({"role": Role.USER.value, "content": "\n\n".join(content)})  # "prefix text\n\ndocument"
    response = []

    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    convert_videos = partial(_convert_videos, dataset_attr=dataset_attr, data_args=data_args)
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": "",
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output
