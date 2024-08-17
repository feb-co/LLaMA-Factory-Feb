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


import os
import sys
from typing import TYPE_CHECKING, Dict, Literal, Optional, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers.utils.versions import require_version

from ..extras.constants import FILEEXT2TYPE
from ..extras.logging import get_logger
from ..extras.misc import has_tokenized_data
from .data_utils import merge_dataset, split_dataset
from .preprocess import get_preprocess_and_print_func
from .template import get_template_and_fix_tokenizer

from .aligner_feb import align_dataset
from .parser_feb import get_dataset_list

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import (
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
    )

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .template import Template
    
    from .parser_feb import DatasetAttr


logger = get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File {} not found.".format(local_path))

        if data_path is None:
            raise ValueError(
                "Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys()))
            )
    else:
        raise NotImplementedError(
            "Unknown load type: {}.".format(dataset_attr.load_from)
        )

    if dataset_attr.load_from == "ms_hub":
        require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")
        from modelscope import MsDataset
        from modelscope.utils.config_ds import MS_DATASETS_CACHE

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
            trust_remote_code=True
        )

    if data_args.streaming and (
        dataset_attr.load_from == "file"
    ):  # faster than specifying streaming=True
        dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

    # samplse dataset
    if dataset_attr.samples_ratio is not None and not data_args.streaming:
        target_num = int(len(dataset) * dataset_attr.samples_ratio)
        indexes = np.random.permutation(len(dataset))[:target_num]
        remaining_num = target_num - len(indexes)
        if remaining_num > 0:
            expand_indexes = np.random.choice(len(dataset), remaining_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == target_num, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info(
            "Sampled {} examples from dataset {}.".format(
                target_num, dataset_attr
            )
        )

    # truncate dataset
    if data_args.max_samples is not None:
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pretrain", "conversation", "instruction"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    if dataset is None:
        return None

    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args,
        stage,
        template,
        tokenizer,
        processor,
        do_generate=(training_args.predict_with_generate and is_eval),
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache)
            or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        preprocess_func, batched=True, remove_columns=column_names, **kwargs
    )

    if training_args.should_log:
        try:
            print(f"{stage} eval example:" if is_eval else f"{stage} training example:", flush=True)
            print_function(next(iter(dataset)))
        except StopIteration:
            raise RuntimeError(
                "Cannot find valid samples, check `data/README.md` for the data format."
            )

    return dataset


def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    template = get_template_and_fix_tokenizer(
        tokenizer, data_args.template, data_args.tool_format
    )
    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    # Load tokenized dataset
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning(
                "Loading dataset from disk will ignore other data arguments."
            )
            dataset_dict: "DatasetDict" = load_from_disk(data_args.tokenized_path)
            logger.info(
                "Loaded tokenized dataset from {}.".format(data_args.tokenized_path)
            )

            dataset_module: Dict[str, "Dataset"] = {}
            if "train" in dataset_dict:
                dataset_module["train_dataset"] = dataset_dict["train"]
            if "validation" in dataset_dict:
                dataset_module["eval_dataset"] = dataset_dict["validation"]

            if data_args.streaming:
                dataset_module = {
                    k: v.to_iterable_dataset() for k, v in dataset_module.items()
                }

            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    dataset_list = []
    for dataset_attr in get_dataset_list(data_args.dataset, data_args.dataset_dir):
        with training_args.main_process_first(
            desc=f"load dataset: {dataset_attr.dataset_name}"
        ):
            if (dataset_attr.stage == "rm" and dataset_attr.ranking is False) or (
                dataset_attr.stage != "rm" and dataset_attr.ranking is True
            ):
                raise ValueError(
                    f"The dataset ({dataset_attr.dataset_name}) is not applicable in the current training stage ({dataset_attr.stage})."
                )
            sub_dataset = _load_single_dataset(
                dataset_attr, model_args, data_args, training_args
            )

        with training_args.main_process_first(
            desc=f"pre-process dataset: {dataset_attr.dataset_name}"
        ):
            sub_dataset = _get_preprocessed_dataset(
                sub_dataset,
                data_args,
                training_args,
                dataset_attr.stage,
                template,
                tokenizer,
                processor,
                is_eval=False,
            )

        dataset_list.append(sub_dataset)

    dataset = merge_dataset(dataset_list, data_args, seed=training_args.seed)

    if data_args.val_size > 1e-6:
        dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
    else:
        dataset_dict = {}
        if dataset is not None:
            if data_args.streaming:
                dataset = dataset.shuffle(
                    buffer_size=data_args.buffer_size, seed=training_args.seed
                )

            dataset_dict["train"] = dataset

        dataset_dict = DatasetDict(dataset_dict)

    if data_args.tokenized_path is not None:
        dataset_dict.save_to_disk(data_args.tokenized_path)
        logger.info(
            "Tokenized dataset saved at {}.".format(data_args.tokenized_path)
        )
        logger.info(
            "Please restart the training with `tokenized_path: {}`.".format(
                data_args.tokenized_path
            )
        )

        sys.exit(0)

    dataset_module = {}
    if "train" in dataset_dict:
        dataset_module["train_dataset"] = dataset_dict["train"]

    return dataset_module
