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
import glob
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk, set_caching_enabled
from transformers.utils.versions import require_version

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import has_tokenized_data
from .data_utils import merge_dataset, split_dataset
from .preprocess import get_preprocess_and_print_func

from .aligner_feb import align_dataset
from .parser_feb import get_dataset_list

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .template import Template
    
    from .parser_feb import DatasetAttr


logger = logging.get_logger(__name__)


def _set_start_method():
    try:
        import multiprocess
        if multiprocess.get_start_method() != 'spawn':
            multiprocess.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


_set_start_method()


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
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
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    elif dataset_attr.load_from == "arrow":
        data_files = []
        local_path = os.path.join(dataset_attr.dataset_name, dataset_attr.split)
        if os.path.isdir(local_path):  # is directory
            data_files += list(glob.glob(os.path.join(local_path, "*.arrow")))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        require_version("openmind>=0.8.0", "To fix: pip install openmind>=0.8.0")
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=data_args.streaming,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
        )

    # samplse dataset
    if dataset_attr.samples_ratio is not None and not data_args.streaming:
        target_num = int(len(dataset) * dataset_attr.samples_ratio)
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        remaining_num = target_num - len(indexes)
        if remaining_num > 0:
            expand_indexes = np.random.choice(len(dataset), remaining_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == target_num, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {target_num} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pretrain", "conversation", "instruction"],
    formatting: Literal["alpaca", "sharegpt", "document", "audio", "audio_arrow_asr", "audio_arrow_tts"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None

    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        preprocess_func,
        batched=True,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print(f"{stage} eval example:" if is_eval else f"{stage} training example:", flush=True)
            print_function(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """
    # Load tokenized dataset
    if data_args.tokenized_path is not None:
        if isinstance(data_args.tokenized_path, str):
            if has_tokenized_data(data_args.tokenized_path):
                logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
                tokenized_data: Union["Dataset", "DatasetDict"] = load_from_disk(data_args.tokenized_path)
                logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")

                dataset_module: Dict[str, "Dataset"] = {}
                if isinstance(tokenized_data, DatasetDict):
                    if "train" in tokenized_data:
                        dataset_module["train_dataset"] = tokenized_data["train"]

                    if "validation" in tokenized_data:
                        dataset_module["eval_dataset"] = tokenized_data["validation"]

                else:  # Dataset
                    dataset_module["train_dataset"] = tokenized_data

                if data_args.streaming:
                    dataset_module = {k: v.to_iterable_dataset() for k, v in dataset_module.items()}

                return dataset_module
        elif isinstance(data_args.tokenized_path, list):
            train_dataset_list = []
            for tokenizer_dir in data_args.tokenized_path:
                if has_tokenized_data(tokenizer_dir):
                    logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
                    tokenized_data: Union["Dataset", "DatasetDict"] = load_from_disk(tokenizer_dir)
                    logger.info_rank0(f"Loaded tokenized dataset from {tokenizer_dir}.")

                    if isinstance(tokenized_data, DatasetDict):
                        train_dataset_list.append(tokenized_data["train"])
                    else:
                        train_dataset_list.append(tokenized_data)
                else:
                    raise ValueError(f"{tokenizer_dir} not exist, list tokenized_path only support for exist dataset path.")

            train_dataset = merge_dataset(train_dataset_list, data_args, seed=training_args.seed)
            train_dataset = train_dataset.shuffle(seed=training_args.seed)
            dataset_module: Dict[str, "Dataset"] = {}
            dataset_module["train_dataset"] = train_dataset
            if data_args.streaming:
                dataset_module = {k: v.to_iterable_dataset() for k, v in dataset_module.items()}
            return dataset_module
        else:
            raise ValueError("not support other type tokenized_path.")

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
                dataset_attr.formatting,
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
            dataset = dataset.shuffle(seed=training_args.seed)
            dataset_dict["train"] = dataset

        dataset_dict = DatasetDict(dataset_dict)

    if data_args.tokenized_path is not None:
        dataset_dict.save_to_disk(data_args.tokenized_path)
        logger.info_rank0(f"Tokenized dataset saved at {data_args.tokenized_path}.")
        logger.info_rank0(f"Please restart the training with `tokenized_path: {data_args.tokenized_path}`.")

        sys.exit(0)

    dataset_module = {}
    if "train" in dataset_dict:
        dataset_module["train_dataset"] = dataset_dict["train"]

    return dataset_module
