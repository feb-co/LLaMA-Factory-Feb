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

from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple


from .pairwise import (
    TextPairwiseDatasetProcessor,
    AudioPairwiseDatasetProcessor,
)
from .pretrain import PretrainDatasetProcessor
from .supervised_text import (
    ConversationDatasetProcessor,
    PackedConversationDatasetProcessor,
    InstructionDatasetProcessor
)
from .process_audio import (
    AvatarAudioDatasetProcessor,
    PackedAvatarAudioDatasetProcessor
)



if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template_feb import TemplateFeb



def get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["dpo", "dpo_audio", "pretrain", "conversation", "instruction", "avatar_audio"],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
) -> Tuple[Callable, Callable]:
    if stage == "dpo":
        dataset_processor_class = TextPairwiseDatasetProcessor
    elif stage == "dpo_audio":
        dataset_processor_class = AudioPairwiseDatasetProcessor
    elif stage == "pretrain":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "conversation":
        if data_args.packing:
            dataset_processor_class = PackedConversationDatasetProcessor
        else:
            dataset_processor_class = ConversationDatasetProcessor
    elif stage == "instruction":
        dataset_processor_class = InstructionDatasetProcessor
    elif stage == "avatar_audio":
        if data_args.packing:
            dataset_processor_class = PackedAvatarAudioDatasetProcessor
        else:
            dataset_processor_class = AvatarAudioDatasetProcessor
    else:
        raise NotImplementedError

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)
