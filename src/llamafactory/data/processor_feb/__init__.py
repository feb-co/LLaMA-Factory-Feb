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

from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple


from .pairwise import (
    TextPairwiseDatasetProcessor,
    AudioPairwiseDatasetProcessor,
)
from .process_text import (
    preprocess_packed_conversation_dataset,
    preprocess_conversation_dataset,
    print_conversation_dataset_example,
    preprocess_instruction_dataset,
    preprocess_pretrain_dataset,
    print_pretrain_dataset_example,
)
from .process_audio import (
    preprocess_avater_audio_dataset,
    preprocess_packed_avater_audio_dataset,
    print_avater_audio_dataset_example
)



if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template_feb import TemplateFeb



def get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["dpo", "dpo_audio", "pretrain", "conversation", "instruction", "avater_audio"],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
) -> Tuple[Callable, Callable]:
    if stage == "dpo":
        dataset_processor_class = TextPairwiseDatasetProcessor
    elif stage == "dpo_audio":
        dataset_processor_class = AudioPairwiseDatasetProcessor
    elif stage == "conversation":
        if data_args.packing:
            preprocess_func = partial(
                preprocess_packed_conversation_dataset,
                template=template,
                tokenizer=tokenizer,
                data_args=data_args,
            )
        else:
            preprocess_func = partial(
                preprocess_conversation_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )

        print_function = partial(
            print_conversation_dataset_example, tokenizer=tokenizer
        )
    elif stage == "pretrain":
        preprocess_func = partial(
            preprocess_pretrain_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        print_function = partial(
            print_pretrain_dataset_example, tokenizer=tokenizer
        )
    elif stage == "instruction":
        preprocess_func = partial(
            preprocess_instruction_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(
            print_conversation_dataset_example, tokenizer=tokenizer
        )
    elif stage == "avater_audio":
        if data_args.packing:
            preprocess_func = partial(
                preprocess_packed_avater_audio_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        else:
            preprocess_func = partial(
                preprocess_avater_audio_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        print_function = partial(
            print_avater_audio_dataset_example, tokenizer=tokenizer
        )
    else:
        raise NotImplementedError

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)
