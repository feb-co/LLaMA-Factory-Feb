#
# Created on Sat Jan 01 2025
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from typing_extensions import override

from ..extras import logging
from .data_utils import Role
from .formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from .mm_plugin import get_mm_plugin


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..hparams import DataArguments
    from .formatter import SLOTS, Formatter
    from .mm_plugin import BasePlugin
    from .tool_utils import FunctionCall


logger = logging.get_logger(__name__)


@dataclass
class TemplateFeb:
    name: str
    format_user: "Formatter"
    format_user_prefix: "Formatter"
    format_user_suffix: "Formatter"
    format_user_audio_prefix: "Formatter"
    format_user_audio_suffix: "Formatter"
    format_thought: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: list[str]
    thought_words: tuple[str, str]
    efficient_eos: bool
    replace_eos: bool
    replace_jinja_template: bool
    mm_plugin: "BasePlugin"

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        r"""Return a single pair of token ids representing prompt and response respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        response_ids = encoded_messages[-1]
        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def encode_multiturn_with_longthought(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int], list[int]]]:
        r"""Return multiple pairs of token ids representing prompts, thoughts and responses respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1], encoded_messages[i + 2]) for i in range(0, len(encoded_messages), 3)]

    def encode_system(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        r"""Return a token ids representing system prompt."""
        if len(system) == 0 and len(tools) == 0:
            return []

        elements = []
        elements += self.format_prefix.apply()
        tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
        elements += self.format_system.apply(content=(system + tool_text))
        system_message = self._convert_txt_elements_to_ids(tokenizer, elements)
        return system_message

    def encode_instruction(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
    ) -> list[tuple[list[int], list[int]]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        if messages[0]['role'] == Role.SYSTEM.value:
            system = messages[0]['content']
        else:
            system = None
        encoded_messages = self._encode(tokenizer, messages, system, None)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def encode_avatar_audio(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
    ) -> list[tuple[dict, dict]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        prompt_pairs = []
        for i, message in enumerate(messages):
            if message["role"] == Role.USER.value:
                elements = self.format_user.apply(content=message["content"], idx=str(i // 2))
                token_ids = self._convert_txt_elements_to_ids(tokenizer, elements)
                token_elem = {"token_ids": token_ids}
            elif message["role"] == Role.USER_AUDIO.value:
                audio_features = []
                audio_positions = []
                token_ids = tokenizer.encode(
                    "".join(self.format_user_prefix.apply()),
                    add_special_tokens=False
                )
                for elem_idx, elem in enumerate(message["content"]):
                    if elem["type"] == "text":
                        token_ids += tokenizer.encode(elem["text"], add_special_tokens=False)
                    elif elem["type"] == "audio":
                        if elem_idx==0 or message["content"][elem_idx-1]["type"] != "audio":
                            token_ids += tokenizer.encode(
                                "".join(self.format_user_audio_prefix.apply()),
                                add_special_tokens=False
                            )

                        audio_length, audio_feature = tokenizer.encode_audio_feature(elem)
                        audio_features.append(audio_feature)
                        audio_positions.append([len(token_ids), audio_length])
                        token_ids += [tokenizer.pad_token_id] * audio_length

                        if elem_idx==len(message["content"])-1 or message["content"][elem_idx+1]["type"] != "audio":
                            token_ids += tokenizer.encode(
                                "".join(self.format_user_audio_suffix.apply()),
                                add_special_tokens=False
                            )
                    else:
                        raise NotImplementedError(f"Unexpected data type: {elem['type']} for role: {Role.USER_AUDIO.value}")
                token_ids += tokenizer.encode(
                    "".join(self.format_user_suffix.apply()),
                    add_special_tokens=False
                )
                token_elem = {"token_ids": token_ids, "audio_features": audio_features, "audio_positions": audio_positions}
            elif message["role"] == Role.ASSISTANT.value:
                elements = self.format_assistant.apply(content=message["content"])
                token_ids = self._convert_txt_elements_to_ids(tokenizer, elements)
                token_elem = {"token_ids": token_ids}
            elif message["role"] == Role.ASSISTANT_AUDIO.value:
                text_elements = self.format_assistant.apply(content=message["content"])
                token_ids = self._convert_txt_elements_to_ids(tokenizer, text_elements)
                response_encode = tokenizer.encode(text=None, audio_signal=message["audios"], add_special_tokens=False)
                token_elem = {"token_ids": token_ids, "audio_codes": response_encode[1]}
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            prompt_pairs.append(token_elem)

        return [(prompt_pairs[i], prompt_pairs[i + 1]) for i in range(0, len(prompt_pairs), 2)]

    def extract_tool(self, content: str) -> Union[str, list["FunctionCall"]]:
        r"""Extract tool message."""
        return self.format_tools.extract(content)

    def get_stop_token_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Return stop token ids."""
        stop_token_ids = {tokenizer.eos_token_id}
        for token in self.stop_words:
            stop_token_ids.add(tokenizer.convert_tokens_to_ids(token))

        return list(stop_token_ids)

    def _convert_txt_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> list[int]:
        r"""Convert elements to token ids."""
        bos_token_id = tokenizer.bos_token_id if getattr(tokenizer, "bos_token_id", None) else tokenizer.text_tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) else tokenizer.text_tokenizer.eos_token_id

        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and bos_token_id is not None:
                    token_ids += [bos_token_id]
                elif "eos_token" in elem and eos_token_id is not None:
                    token_ids += [eos_token_id]
            else:
                raise ValueError(f"Input must be string, set[str] or dict[str, str], got {type(elem)}")

        return token_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        r"""Encode formatted inputs to pairs of token ids.

        Turn 0: prefix + system + query        resp
        Turn t: query                          resp.
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                if system or tools:
                    elements += self.format_prefix.apply()
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.THINK.value:
                elements += self.format_thought.apply(content=message["content"])
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            elif message["role"] == Role.MASK.value:
                elements += self.format_assistant.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_txt_elements_to_ids(tokenizer, elements))

        return encoded_messages

    @staticmethod
    def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
        r"""Add or replace eos token to the tokenizer."""
        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

        if is_added:
            logger.info_rank0(f"Add eos token: {tokenizer.eos_token}.")
        else:
            logger.info_rank0(f"Replace eos token: {tokenizer.eos_token}.")

        if num_added_tokens > 0:
            logger.warning_rank0("New tokens have been added, make sure `resize_vocab` is True.")

    def fix_special_tokens(self, tokenizer: "PreTrainedTokenizer") -> None:
        r"""Add eos token and pad token to the tokenizer."""
        stop_words = self.stop_words
        if self.replace_eos:
            if not stop_words:
                raise ValueError("Stop words are required to replace the EOS token.")

            self._add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
            stop_words = stop_words[1:]

        if tokenizer.eos_token_id is None:
            self._add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info_rank0(f"Add pad token: {tokenizer.pad_token}")

        if stop_words:
            num_added_tokens = tokenizer.add_special_tokens(
                dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
            )
            logger.info_rank0("Add {} to stop words.".format(",".join(stop_words)))
            if num_added_tokens > 0:
                logger.warning_rank0("New tokens have been added, make sure `resize_vocab` is True.")

    @staticmethod
    def _jinja_escape(content: str) -> str:
        r"""Escape single quotes in content."""
        return content.replace("'", r"\'")

    @staticmethod
    def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
        r"""Convert slots to jinja template."""
        slot_items = []
        for slot in slots:
            if isinstance(slot, str):
                slot_pieces = slot.split("{{content}}")
                if slot_pieces[0]:
                    slot_items.append("'" + TemplateFeb._jinja_escape(slot_pieces[0]) + "'")
                if len(slot_pieces) > 1:
                    slot_items.append(placeholder)
                    if slot_pieces[1]:
                        slot_items.append("'" + TemplateFeb._jinja_escape(slot_pieces[1]) + "'")
            elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
                if "bos_token" in slot and tokenizer.bos_token_id is not None:
                    slot_items.append("'" + tokenizer.bos_token + "'")
                elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                    slot_items.append("'" + tokenizer.eos_token + "'")
            elif isinstance(slot, dict):
                raise ValueError("Dict is not supported.")

        return " + ".join(slot_items)

    def _get_jinja_template(self, tokenizer: "PreTrainedTokenizer") -> str:
        r"""Return the jinja template."""
        prefix = self._convert_slots_to_jinja(self.format_prefix.apply(), tokenizer)
        system = self._convert_slots_to_jinja(self.format_system.apply(), tokenizer, placeholder="system_message")
        user = self._convert_slots_to_jinja(self.format_user.apply(), tokenizer)
        assistant = self._convert_slots_to_jinja(self.format_assistant.apply(), tokenizer)
        jinja_template = ""
        if prefix:
            jinja_template += "{{ " + prefix + " }}"

        if self.default_system:
            jinja_template += "{% set system_message = '" + self._jinja_escape(self.default_system) + "' %}"

        jinja_template += (
            "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
            "{% if system_message is defined %}{{ " + system + " }}{% endif %}"
            "{% for message in loop_messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ " + user + " }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ " + assistant + " }}"
            "{% endif %}"
            "{% endfor %}"
        )
        return jinja_template

    def fix_jinja_template(self, tokenizer: "PreTrainedTokenizer") -> None:
        r"""Replace the jinja template in the tokenizer."""
        if tokenizer.chat_template is None or self.replace_jinja_template:
            try:
                tokenizer.chat_template = self._get_jinja_template(tokenizer)
            except ValueError as e:
                logger.info_rank0(f"Cannot add this chat template to tokenizer: {e}.")

    @staticmethod
    def _convert_slots_to_ollama(
        slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content"
    ) -> str:
        r"""Convert slots to ollama template."""
        slot_items = []
        for slot in slots:
            if isinstance(slot, str):
                slot_pieces = slot.split("{{content}}")
                if slot_pieces[0]:
                    slot_items.append(slot_pieces[0])
                if len(slot_pieces) > 1:
                    slot_items.append("{{ " + placeholder + " }}")
                    if slot_pieces[1]:
                        slot_items.append(slot_pieces[1])
            elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
                if "bos_token" in slot and tokenizer.bos_token_id is not None:
                    slot_items.append(tokenizer.bos_token)
                elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                    slot_items.append(tokenizer.eos_token)
            elif isinstance(slot, dict):
                raise ValueError("Dict is not supported.")

        return "".join(slot_items)

    def _get_ollama_template(self, tokenizer: "PreTrainedTokenizer") -> str:
        r"""Return the ollama template."""
        prefix = self._convert_slots_to_ollama(self.format_prefix.apply(), tokenizer)
        system = self._convert_slots_to_ollama(self.format_system.apply(), tokenizer, placeholder=".System")
        user = self._convert_slots_to_ollama(self.format_user.apply(), tokenizer, placeholder=".Content")
        assistant = self._convert_slots_to_ollama(self.format_assistant.apply(), tokenizer, placeholder=".Content")
        return (
            f"{prefix}{{{{ if .System }}}}{system}{{{{ end }}}}"
            f"""{{{{ range .Messages }}}}{{{{ if eq .Role "user" }}}}{user}"""
            f"""{{{{ else if eq .Role "assistant" }}}}{assistant}{{{{ end }}}}{{{{ end }}}}"""
        )

    def get_ollama_modelfile(self, tokenizer: "PreTrainedTokenizer") -> str:
        r"""Return the ollama modelfile.

        TODO: support function calling.
        """
        modelfile = "# ollama modelfile auto-generated by llamafactory\n\n"
        modelfile += f'FROM .\n\nTEMPLATE """{self._get_ollama_template(tokenizer)}"""\n\n'

        if self.default_system:
            modelfile += f'SYSTEM """{self.default_system}"""\n\n'

        for stop_token_id in self.get_stop_token_ids(tokenizer):
            modelfile += f'PARAMETER stop "{tokenizer.convert_ids_to_tokens(stop_token_id)}"\n'

        modelfile += "PARAMETER num_ctx 4096\n"
        return modelfile


TEMPLATES: dict[str, "TemplateFeb"] = {}


def register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_user_prefix: Optional["Formatter"] = None,
    format_user_suffix: Optional["Formatter"] = None,
    format_user_audio_prefix: Optional["Formatter"] = None,
    format_user_audio_suffix: Optional["Formatter"] = None,
    format_thought: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: Optional[list[str]] = None,
    thought_words: Optional[tuple[str, str]] = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    replace_jinja_template: bool = False,
    mm_plugin: "BasePlugin" = get_mm_plugin(name="base"),
    template_class: type["TemplateFeb"] = TemplateFeb,
) -> None:
    r"""
    The parameter `efficient_eos` is used to determine whether eos needs to be added to the input. If not, it is True, otherwise it is False.

    Register a chat template.

    To add the following chat template:
    ```
    <s><user>user prompt here
    <model>model response here</s>
    <user>user prompt here
    <model>model response here</s>
    ```

    The corresponding code should be:
    ```
    register_template(
        name="custom",
        format_user=StringFormatter(slots=["<user>{{content}}\n<model>"]),
        format_assistant=StringFormatter(slots=["{{content}}</s>\n"]),
        format_prefix=EmptyFormatter("<s>"),
    )
    ```
    """
    if name in TEMPLATES:
        raise ValueError(f"Template {name} already exists.")

    default_slots = ["{{content}}"] if efficient_eos else ["{{content}}", {"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_user_prefix_formatter = EmptyFormatter()
    default_user_suffix_formatter = EmptyFormatter()
    default_user_audio_prefix_formatter = EmptyFormatter()
    default_user_audio_suffix_formatter = EmptyFormatter()
    default_thought_formatter = StringFormatter(slots=default_slots)
    default_assistant_formatter = StringFormatter(slots=default_slots)
    default_function_formatter = FunctionFormatter(slots=default_slots, tool_format="default")
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        name=name,
        format_user=format_user or default_user_formatter,
        format_user_prefix=format_user_prefix or default_user_prefix_formatter,
        format_user_suffix=format_user_suffix or default_user_suffix_formatter,
        format_user_audio_prefix=format_user_audio_prefix or default_user_audio_prefix_formatter,
        format_user_audio_suffix=format_user_audio_suffix or default_user_audio_suffix_formatter,
        format_thought=format_thought or default_thought_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words or [],
        thought_words=thought_words or ("<think>", "</think>"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
        mm_plugin=mm_plugin,
    )


def parse_template(tokenizer: "PreTrainedTokenizer") -> "TemplateFeb":
    r"""Extract a chat template from the tokenizer."""

    def find_diff(short_str: str, long_str: str) -> str:
        i, j = 0, 0
        diff = ""
        while i < len(short_str) and j < len(long_str):
            if short_str[i] == long_str[j]:
                i += 1
                j += 1
            else:
                diff += long_str[j]
                j += 1

        return diff

    prefix = tokenizer.decode(tokenizer.encode(""))

    messages = [{"role": "system", "content": "{{content}}"}]
    system_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[len(prefix) :]

    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "{{content}}"}]
    user_slot_empty_system = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot_empty_system = user_slot_empty_system[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}]
    user_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot = user_slot[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}, {"role": "assistant", "content": "{{content}}"}]
    assistant_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    assistant_slot = assistant_slot[len(prefix) + len(user_slot) :]

    if len(user_slot) > len(user_slot_empty_system):
        default_system = find_diff(user_slot_empty_system, user_slot)
        sole_system = system_slot.replace("{{content}}", default_system, 1)
        user_slot = user_slot[len(sole_system) :]
    else:  # if defaut_system is empty, user_slot_empty_system will be longer than user_slot
        default_system = ""

    return TemplateFeb(
        format_user=StringFormatter(slots=[user_slot]),
        format_assistant=StringFormatter(slots=[assistant_slot]),
        format_system=StringFormatter(slots=[system_slot]),
        format_function=FunctionFormatter(slots=[assistant_slot], tool_format="default"),
        format_observation=StringFormatter(slots=[user_slot]),
        format_tools=ToolFormatter(tool_format="default"),
        format_prefix=EmptyFormatter(slots=[prefix]) if prefix else EmptyFormatter(),
        default_system=default_system,
        stop_words=[],
        thought_words=("<think>", "</think>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        mm_plugin=get_mm_plugin(name="base"),
    )


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", data_args: "DataArguments") -> "TemplateFeb":
    r"""Get chat template and fixes the tokenizer."""
    if data_args.template is None:
        if isinstance(tokenizer.chat_template, str):
            logger.warning_rank0("`template` was not specified, try parsing the chat template from the tokenizer.")
            template = parse_template(tokenizer)
        else:
            logger.warning_rank0("`template` was not specified, use `empty` template.")
            template = TEMPLATES["empty"]  # placeholder
    else:
        if data_args.template not in TEMPLATES:
            raise ValueError(f"Template {data_args.template} does not exist.")

        template = TEMPLATES[data_args.template]

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    if data_args.tool_format is not None:
        logger.info_rank0(f"Using tool format: {data_args.tool_format}.")
        default_slots = ["{{content}}"] if template.efficient_eos else ["{{content}}", {"eos_token"}]
        template.format_function = FunctionFormatter(slots=default_slots, tool_format=data_args.tool_format)
        template.format_tools = ToolFormatter(tool_format=data_args.tool_format)

    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)
    return template


register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
            )
        ]
    ),
    format_user_prefix=EmptyFormatter(
        slots=[
            "<|start_header_id|>user<|end_header_id|>\n\n"
        ]
    ),
    format_user_suffix=EmptyFormatter(
        slots=[
            "<|eot_id|>"
        ]
    ),
    format_user_audio_prefix=EmptyFormatter(
        slots=[
            "<audio>"
        ]
    ),
    format_user_audio_suffix=EmptyFormatter(
        slots=[
            "</audio>"
        ]
    ),
    format_thought=StringFormatter(
        slots=["<|start_header_id|>thought<|end_header_id|>\n\n{{content}}", {"eos_token"}]
    ),
    format_assistant=StringFormatter(
        slots=["<|start_header_id|>assistant<|end_header_id|>\n\n{{content}}", {"eos_token"}]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_function=FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
            )
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
    replace_jinja_template=True,
)
