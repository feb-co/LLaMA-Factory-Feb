#
# Created on Sun Aug 04 2024
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


import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

from transformers.utils import cached_file

from ..extras.constants import DATA_CONFIG
from ..extras.misc import use_modelscope


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "script", "file"]
    dataset_name: str
    stage: Literal["pretrain", "conversation", "instruction"] = "conversation"
    formatting: Literal["alpaca", "sharegpt", "document"] = "sharegpt"
    ranking: bool = False

    # extra configs
    subset: Optional[str] = None
    split: str = "train"
    folder: Optional[str] = None
    samples_ratio: Optional[float] = None

    # common columns
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None

    # rlhf columns
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
    assistant_tag: Optional[str] = "assistant"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"
    mask_tag: Optional[str] = "mask"

    # document columns
    prefix: Optional[str] = "prefix_text"
    document: Optional[str] = "document"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(
        self, key: str, obj: Dict[str, Any], default: Optional[Any] = None
    ) -> None:
        setattr(self, key, obj.get(key, default))


def get_dataset_list(
    dataset_names: Optional[Sequence[str]], dataset_dir: str
) -> List["DatasetAttr"]:
    r"""
    Gets the attributes of the datasets.
    """
    if dataset_names is None:
        dataset_names = []

    if dataset_dir == "ONLINE":
        dataset_info = None
    else:
        if dataset_dir.startswith("REMOTE:"):
            config_path = cached_file(
                path_or_repo_id=dataset_dir[7:],
                filename=DATA_CONFIG,
                repo_type="dataset",
            )
        else:
            config_path = os.path.join(dataset_dir, DATA_CONFIG)

        try:
            with open(config_path, "r") as f:
                dataset_info = json.load(f)
        except Exception as err:
            if len(dataset_names) != 0:
                raise ValueError(
                    "Cannot open {} due to {}.".format(config_path, str(err))
                )

            dataset_info = None

    dataset_list: List["DatasetAttr"] = []
    for name in dataset_names:
        if dataset_info is None:  # dataset_dir is ONLINE
            load_from = "ms_hub" if use_modelscope() else "hf_hub"
            dataset_attr = DatasetAttr(load_from, dataset_name=name)
            dataset_list.append(dataset_attr)
            continue

        if name not in dataset_info:
            raise ValueError("Undefined dataset {} in {}.".format(name, DATA_CONFIG))

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]

        if has_hf_url or has_ms_url:
            if (use_modelscope() and has_ms_url) or (not has_hf_url):
                dataset_attr = DatasetAttr(
                    "ms_hub", dataset_name=dataset_info[name]["ms_hub_url"]
                )
            else:
                dataset_attr = DatasetAttr(
                    "hf_hub", dataset_name=dataset_info[name]["hf_hub_url"]
                )
        elif "script_url" in dataset_info[name]:
            dataset_attr = DatasetAttr(
                "script", dataset_name=dataset_info[name]["script_url"]
            )
        else:
            dataset_attr = DatasetAttr(
                "file", dataset_name=dataset_info[name]["file_name"]
            )

        dataset_attr.set_attr("stage", dataset_info[name])
        dataset_attr.set_attr("formatting", dataset_info[name], default="sharegpt")
        dataset_attr.set_attr("ranking", dataset_info[name], default=False)
        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("split", dataset_info[name], default="train")
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("samples_ratio", dataset_info[name])

        if "columns" in dataset_info[name]:
            column_names = [
                "system",
                "tools",
                "images",
                "chosen",
                "rejected",
                "kto_tag",
            ]
            if dataset_attr.formatting == "alpaca":
                column_names.extend(["prompt", "query", "response", "history"])
            elif dataset_attr.formatting == "document":
                column_names.extend(["prefix", "document"])
            else:
                column_names.extend(["messages"])

            for column_name in column_names:
                if column_name in dataset_info[name]["columns"]:
                    dataset_attr.set_attr(column_name, dataset_info[name]["columns"])

        if dataset_attr.formatting == "sharegpt" and "tags" in dataset_info[name]:
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
                if tag in dataset_info[name]["tags"]:
                    dataset_attr.set_attr(tag, dataset_info[name]["tags"])

        dataset_list.append(dataset_attr)

    return dataset_list
