#
# Created on Sun Aug 04 2024
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
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from huggingface_hub import hf_hub_download

from ..extras.constants import DATA_CONFIG
from ..extras.misc import use_modelscope, use_openmind


@dataclass
class DatasetAttr:
    r"""Dataset attributes."""

    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "om_hub", "script", "file", "arrow"]
    dataset_name: str
    dataset_key: str = None
    stage: Literal["dpo", "dpo_audio", "pretrain", "conversation", "instruction", "avatar_audio"] = "conversation"
    formatting: Literal["alpaca", "sharegpt", "document", "longthought", "audio", "audio_arrow_asr", "audio_arrow_tts"] = "sharegpt"
    ranking: bool = False

    # extra configs
    subset: Optional[str] = None
    split: str = "train"
    folder: Optional[str] = None
    samples_ratio: Optional[float] = None

    # common columns
    system: Optional[str] = None
    system_list: Optional[list] = None
    style_list: Optional[list] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    audios: Optional[str] = None

    # dpo columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None

    # alpaca columns
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None

    # sharegpt columns
    messages: Optional[str] = "conversations"

    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "user"
    think_tag: Optional[str] = "think"
    user_audio_tag: Optional[str] = "user_audio"
    assistant_tag: Optional[str] = "assistant"
    assistant_audio_tag: Optional[str] = "assistant_audio"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"
    mask_tag: Optional[str] = "mask"

    # document columns
    prefix: Optional[str] = "prefix_text"
    document: Optional[str] = "document"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))

    def join(self, attr: dict[str, Any]) -> None:
        self.set_attr("stage", attr)
        self.set_attr("formatting", attr, default="sharegpt")
        self.set_attr("ranking", attr, default=False)
        self.set_attr("subset", attr)
        self.set_attr("split", attr, default="train")
        self.set_attr("folder", attr)
        self.set_attr("samples_ratio", attr)

        if "columns" in attr:
            column_names = ["prompt", "query", "response", "history", "messages", "system", "tools"]
            column_names += ["images", "videos", "audios", "chosen", "rejected", "kto_tag"]
            if self.formatting == "document":
                column_names.extend(["prefix", "document"])
            elif "audio" in self.formatting:
                column_names.extend(["system_list"])
                column_names.extend(["style_list"])

            for column_name in column_names:
                if column_name in attr["columns"]:
                    self.set_attr(column_name, attr["columns"])

        if self.formatting == "sharegpt" and "tags" in attr:
            tag_names = (
                "role_tag",
                "content_tag",
                "user_tag",
                "assistant_tag",
                "observation_tag",
                "function_tag",
                "system_tag",
            )
            for tag in tag_names:
                if tag in attr["tags"]:
                    self.set_attr(tag, attr["tags"])


def get_dataset_list(dataset_names: Optional[list[str]], dataset_dir: Union[str, dict]) -> list["DatasetAttr"]:
    r"""Get the attributes of the datasets."""
    if dataset_names is None:
        dataset_names = []

    if isinstance(dataset_dir, dict):
        dataset_info = dataset_dir
    elif dataset_dir == "ONLINE":
        dataset_info = None
    else:
        if dataset_dir.startswith("REMOTE:"):
            config_path = hf_hub_download(repo_id=dataset_dir[7:], filename=DATA_CONFIG, repo_type="dataset")
        else:
            config_path = os.path.join(dataset_dir, DATA_CONFIG)

        try:
            with open(config_path) as f:
                dataset_info = json.load(f)
        except Exception as err:
            if len(dataset_names) != 0:
                raise ValueError(f"Cannot open {config_path} due to {str(err)}.")

            dataset_info = None

    dataset_list: list[DatasetAttr] = []
    for name in dataset_names:
        if dataset_info is None:  # dataset_dir is ONLINE
            load_from = "ms_hub" if use_modelscope() else "om_hub" if use_openmind() else "hf_hub"
            dataset_attr = DatasetAttr(load_from, dataset_name=name)
            dataset_list.append(dataset_attr)
            continue

        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}.")

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]
        has_om_url = "om_hub_url" in dataset_info[name]

        if has_hf_url or has_ms_url or has_om_url:
            if has_ms_url and (use_modelscope() or not has_hf_url):
                dataset_attr = DatasetAttr("ms_hub", dataset_name=dataset_info[name]["ms_hub_url"], dataset_key=name)
            elif has_om_url and (use_openmind() or not has_hf_url):
                dataset_attr = DatasetAttr("om_hub", dataset_name=dataset_info[name]["om_hub_url"], dataset_key=name)
            else:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"], dataset_key=name)
        elif "script_url" in dataset_info[name]:
            dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"], dataset_key=name)
        elif "cloud_file_name" in dataset_info[name]:
            dataset_attr = DatasetAttr("cloud_file", dataset_name=dataset_info[name]["cloud_file_name"], dataset_key=name)
        elif "arrow_directory" in dataset_info[name]:
            dataset_attr = DatasetAttr("arrow", dataset_name=dataset_info[name]["arrow_directory"], dataset_key=name)
        else:
            dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"], dataset_key=name)

        dataset_attr.join(dataset_info[name])
        dataset_list.append(dataset_attr)

    return dataset_list
