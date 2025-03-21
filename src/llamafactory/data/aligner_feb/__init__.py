#
# Created on Thu Feb 25 2025
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


from functools import partial
from typing import TYPE_CHECKING, Union

from .aligner_text import convert_alpaca, convert_document, convert_sharegpt, convert_longthought
from .aligner_audio import convert_avater_audio, convert_avater_audio_arrow
from .aligner_audio_arrow_tts import convert_avater_audio_arrow_tts
from ...extras import logging


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ...hparams import DataArguments
    from ..parser_feb import DatasetAttr


logger = logging.get_logger(__name__)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        _system: "..."
        _tools: "...",
        _images: [],
        _videos: [],
    """
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}

    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr, data_args=data_args)
    elif dataset_attr.formatting == "document":
        convert_func = partial(convert_document, dataset_attr=dataset_attr, data_args=data_args)
    elif dataset_attr.formatting == "longthought":
        convert_func = partial(convert_longthought, dataset_attr=dataset_attr, data_args=data_args)
    elif dataset_attr.formatting == "audio":
        convert_func = partial(convert_avater_audio, dataset_attr=dataset_attr, data_args=data_args)
    elif dataset_attr.formatting == "audio_arrow_tts":
        convert_func = partial(convert_avater_audio_arrow_tts, dataset_attr=dataset_attr, data_args=data_args)
    elif dataset_attr.formatting == "audio_arrow_asr":
        convert_func = partial(convert_avater_audio_arrow, dataset_attr=dataset_attr, data_args=data_args)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)

    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    # 根据数据集类型设置批处理参数
    is_batch_dataset = (
        dataset_attr.formatting == "audio_arrow_tts" 
        and any(key in dataset_attr.dataset_key for key in ["dialogue_", "packed_"])
    )
    kwargs.update({
        "batched": is_batch_dataset,
        "batch_size": 16 if is_batch_dataset else None
    })

    return dataset.map(
        convert_func,
        remove_columns=column_names,
        **kwargs,
    )
