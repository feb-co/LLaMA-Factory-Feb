#
# Created on Sun Mar 16 2025
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


SYSTEM_STYLE_PROMPT = [
    """When conversing with the user through voice, adhere to this speech style:\n``````\n{style}\n``````""",
    """In voice dialogue with the user, keep the following vocal style active:\n``````\n{style}\n``````""",
    """As you talk to the user via voice, maintain this manner of speaking:\n``````\n{style}\n``````""",
    """During your voice session with the user, use the style below:\n``````\n{style}\n``````""",
    """While engaging in a voice call with the user, your voice should reflect:\n``````\n{style}\n``````""",
    """Throughout the voice interaction, follow the voice style specified here:\n``````\n{style}\n``````""",
    """In spoken exchange with the user, apply the following voice characteristics:\n``````\n{style}\n``````""",
    """When speaking aloud to the user, stay consistent with this vocal approach:\n``````\n{style}\n``````""",
    """For the voice conversation, keep to this speaking style:\n``````\n{style}\n``````""",
    """As you verbally communicate with the user, mirror this style:\n``````\n{style}\n``````""",
    """Maintain the style below while you talk to the user in voice:\n``````\n{style}\n``````""",
    """Use the following voice persona during the voice chat:\n``````\n{style}\n``````""",
    """During voice-based dialogue, adhere to:\n``````\n{style}\n``````""",
    """When interacting by voice, keep your voice aligned with:\n``````\n{style}\n``````""",
    """Speak to the user using this style throughout the voice session:\n``````\n{style}\n``````""",
    """In every voice exchange with the user, follow this tone:\n``````\n{style}\n``````""",
    """While conducting voice conversations, your speech should exhibit:\n``````\n{style}\n``````""",
    """Ensure the following voice style is present in your spoken replies:\n``````\n{style}\n``````""",
    """Throughout all voice talks with the user, sustain this style:\n``````\n{style}\n``````""",
    """During live voice interaction, present your voice according to:\n``````\n{style}\n``````""",
    """When addressing the user vocally, conform to the style below:\n``````\n{style}\n``````""",
    """As you engage the user in spoken form, use this speech style:\n``````\n{style}\n``````""",
    """For voice responses, model your tone after:\n``````\n{style}\n``````""",
    """In voice communication scenarios, maintain this vocal identity:\n``````\n{style}\n``````""",
    """Use the style below for all voice-based discussions:\n``````\n{style}\n``````""",
    """While conversing aloud, keep your voice styled as follows:\n``````\n{style}\n``````""",
    """In this voice interaction, embody the style shown here:\n``````\n{style}\n``````""",
    """During the spoken session with the user, follow this speaking manner:\n``````\n{style}\n``````""",
    """Maintain the following tone throughout your voice conversation:\n``````\n{style}\n``````""",
    """Your vocal replies during the voice exchange should match:\n``````\n{style}\n``````""",
]


SYSTEM_TTS_PROMPT = [
    """Your task is to repeat the following user's input.""",
    """Your assignment is to echo the following user's input.""",
    """Repeat exactly the user's input provided below.""",
    """Your job is to replicate the user's input exactly as given.""",
    """You are required to repeat the following input from the user.""",
    """Your task is to output the user's input verbatim.""",
    """Your responsibility is to mirror the user's input as it appears.""",
    """Your objective is to duplicate the following user's input.""",
    """Your duty is to repeat exactly what the user has input.""",
    """Echo back the following user's input as your task.""",
    """Your assignment is to reproduce the user's input exactly.""",
    """Repeat the user's input provided below without alterations.""",
    """Your role is to reflect the user's input exactly as given.""",
    """Your task is to copy the following input from the user verbatim.""",
    """You must repeat the user's input exactly as it is presented.""",
    "Please repeat exactly what the user writes below.",
    "Your mission is to return the user's input precisely as entered.",
    "Simply repeat the user's message without adding, removing, or changing anything.",
    "Your duty is to output the user’s words exactly, maintaining every character as written.",
    "Your sole task: replicate the input from the user without any edits or interpretation.",
    "Repeat the following user's input word-for-word and without any deviation.",
    "Kindly reflect the input provided by the user exactly, ensuring no differences.",
    "Your objective is to echo the user's words back exactly, keeping punctuation and spacing identical.",
    "Please provide an exact reproduction of what the user has entered, preserving its original form.",
    "Echo the user's text exactly as received, maintaining perfect fidelity to the input.",
    "Your role is to copy the user's input exactly, respecting original formatting and spacing.",
    "Without interpreting or changing anything, simply repeat the user's message below.",
    "Output exactly what the user has said, making no adjustments to the content or structure.",
    "Repeat the input exactly as it appears, ensuring nothing is added, removed, or altered.",
    "You are assigned to mirror the user's entire input accurately and faithfully, with no omissions."
]


USER_DIALOGUE_PROMPT = [
    "Your task is to randomly generate a sentence or phrase.",
    "Please randomly generate a sentence or phrase.",
    "Generate a random sentence or phrase for me.",
    "Create a random sentence or phrase.",
    "Produce a randomly generated sentence or phrase.",
    "Give me a randomly constructed sentence or phrase.",
    "Come up with a random sentence or phrase.",
    "Offer a random sentence or phrase.",
    "Provide a randomly generated phrase or sentence.",
    "Supply a random sentence or phrase.",
    "Deliver a randomly created sentence or phrase.",
    "Formulate a random sentence or phrase.",
    "Craft a random phrase or sentence",
    "Invent a random sentence or phrase",
    "Concoct a random sentence or phrase",
    "Generate one random sentence or phrase.",
    "Create one random sentence or phrase",
    "Produce one random sentence or phrase",
    "Give me one random sentence or phrase.",
    "Provide one randomly generated sentence or phrase",
    "Offer one random sentence or phrase.",
    "Come up with one random sentence or phrase",
    "Supply one random sentence or phrase",
    "Formulate one random sentence or phrase.",
    "Craft one random sentence or phrase",
    "Invent one random sentence or phrase",
    "Concoct one random sentence or phrase",
    "Generate an arbitrary sentence or phrase",
    "Create an arbitrary sentence or phrase.",
    "Produce an arbitrary sentence or phrase.",
    "Give me an arbitrary sentence or phrase",
    "Provide an arbitrary sentence or phrase",
    "Offer an arbitrary sentence or phrase",
    "Come up with an arbitrary sentence or phrase",
    "Supply an arbitrary sentence or phrase."
]



USER_DIALOGUE_STYLE_PROMPT = [
    """You are using voice to communicate with users. Your task is to randomly generate a sentence or phrase, and meet the following speech style requirements:\n``````\n{style}\n``````""",
    """You are using voice to communicate with users. Please randomly generate a sentence or phrase, and meet the following voice style requirements:\n``````\n{style}\n``````""",
    """You are speaking with the user. Generate a random utterance that conforms to the following speech style guidelines:\n``````\n{style}\n``````""",
    """You are interacting by voice. Provide a randomly composed sentence that adheres to the following vocal style requirements:\n``````\n{style}\n``````""",
    """You are conveying information through speech. Create a random phrase while satisfying these speech style rules:\n``````\n{style}\n``````""",
    """You are communicating verbally. Produce a random expression that matches the following voice style criteria:\n``````\n{style}\n``````""",
    """You are engaging via spoken language. Formulate a random statement under the constraints of the given speech style:\n``````\n{style}\n``````""",
    """You are addressing the user by voice. Craft a random sentence that aligns with these vocal style directions:\n``````\n{style}\n``````""",
    """You are talking aloud to the user. Generate a random phrase obeying the following speech style parameters:\n``````\n{style}\n``````""",
    """You are using audible speech. Randomly produce a sentence that fulfills these vocal style specifications:\n``````\n{style}\n``````""",
    """You are interacting via voice. Randomly generate an utterance following these speech style constraints:\n``````\n{style}\n``````""",
    """You are speaking to the user. Produce a random phrase in accordance with the following voice style rules:\n``````\n{style}\n``````""",
    """You are using spoken words. Generate a random sentence that abides by these speech style guidelines:\n``````\n{style}\n``````""",
    """You are addressing through speech. Craft a random utterance within the given vocal style framework:\n``````\n{style}\n``````""",
    """You are interacting audibly. Form a random phrase respecting these speech style conditions:\n``````\n{style}\n``````""",
    """You are vocalizing to the user. Create a random sentence meeting these voice style requirements:\n``````\n{style}\n``````""",
    """You are conversing by voice. Randomly produce an expression following these speech style guidelines:\n``````\n{style}\n``````""",
    """You are emitting spoken dialogue. Generate a random statement that complies with these vocal style criteria:\n``````\n{style}\n``````""",
    """You are engaging in verbal communication. Create a random phrase under these speech style directives:\n``````\n{style}\n``````""",
    """You are utilizing voice input/output. Produce a random sentence that matches these voice style constraints:\n``````\n{style}\n``````""",
    """You are conveying a message by voice. Generate a random utterance adhering to these speech style rules:\n``````\n{style}\n``````""",
    """You are delivering speech. Randomly craft a phrase in line with the following vocal style requirements:\n``````\n{style}\n``````""",
    """You are providing audio feedback. Create a random sentence that fulfills these speech style specifications:\n``````\n{style}\n``````""",
    """You are giving a verbal response. Formulate a random expression following these voice style constraints:\n``````\n{style}\n``````""",
    """You are producing vocal content. Generate a random phrase in accordance with these speech style guidelines:\n``````\n{style}\n``````""",
    """You are speaking out loud. Create a random utterance that adheres to the following vocal style criteria:\n``````\n{style}\n``````""",
    """You are narrating verbally. Produce a random sentence under these speech style guidelines:\n``````\n{style}\n``````""",
    """You are using vocal expression. Randomly generate a phrase that meets these speech style conditions:\n``````\n{style}\n``````""",
    """You are articulating words aloud. Create a random statement following these vocal style rules:\n``````\n{style}\n``````""",
    """You are communicating orally. Randomly produce a sentence that aligns with these speech style specifications:\n``````\n{style}\n``````""",
    """You are giving spoken instructions. Generate a random phrase that complies with these voice style requirements:\n``````\n{style}\n``````""",
    """You are chatting using speech. Craft a random utterance while satisfying these speech style guidelines:\n``````\n{style}\n``````""",
    """You are interacting via audible dialogue. Formulate a random sentence adhering to these vocal style constraints:\n``````\n{style}\n``````""",
    """You are engaging in spoken interaction. Produce a random phrase that meets the following speech style criteria:\n``````\n{style}\n``````""",
    """You are conveying ideas by voice. Create a random utterance consistent with these vocal style rules:\n``````\n{style}\n``````""",
]


USER_TTS_PROMPT = [
    """Please repeat the following user's input (which may contain some special symbols):\n``````\n{text}\n``````""",
    """Your task is to echo the following user's input (which may include special symbols):\n``````\n{text}\n``````""",
    """Repeat the following user's input exactly as provided, even if it contains special symbols:\n``````\n{text}\n``````""",
    """Output the following user's input precisely as given (special symbols included):\n``````\n{text}\n``````""",
    """Your assignment is to replicate the following user's input (note that it may include special symbols):\n``````\n{text}\n``````""",
    """Please echo back the user's input exactly, including any special symbols:\n``````\n{text}\n``````""",
    """Your job is to reproduce the following user's input verbatim (including special symbols):\n``````\n{text}\n``````""",
    """Repeat exactly the user's input provided below (special symbols must be retained):\n``````\n{text}\n``````""",
    """You are required to repeat the following user's input exactly as given (with special symbols intact):\n``````\n{text}\n``````""",
    """Your task is to mirror the following user's input exactly (special symbols should remain unchanged):\n``````\n{text}\n``````""",
    """Please duplicate the following user's input word-for-word (special symbols are permitted):\n``````\n{text}\n``````""",
    """Repeat the following user's input exactly, including any special symbols:\n``````\n{text}\n``````""",
    """Your assignment is to echo exactly the user's input shown below (special symbols included):\n``````\n{text}\n``````""",
    """Your task is to output the following user's input precisely as provided (note: it may contain special symbols):\n``````\n{text}\n``````""",
    """Repeat exactly the user's input provided below, ensuring that all special symbols are preserved:\n``````\n{text}\n``````""",
    "Kindly return the exact user submission shown below, maintaining every character, punctuation mark, digit, or symbol inside it:\n``````\n{text}\n``````",
    "Your sole duty is to print the user's upcoming text exactly as it appears, without altering capitalization or special characters:\n``````\n{text}\n``````",
    "Please reproduce the user's following entry verbatim, ensuring that all whitespace, line breaks, emojis, and unusual glyphs remain untouched:\n``````\n{text}\n``````",
    "Echo precisely the text supplied by the user below—make no edits, substitutions, insertions, or deletions of any kind:\n``````\n{text}\n``````",
    "You are instructed to output the user's text exactly, including repeated spaces, tabs, line breaks, or other formatting quirks that might exist:\n``````\n{text}\n``````",
    "Return the user's next message unchanged; preserve every accent mark, mathematical operator, bracket, and quotation mark exactly as presented:\n``````\n{text}\n``````",
    "Mirror the forthcoming user text character-for-character; do not interpret, translate, or sanitize any part of it:\n``````\n{text}\n``````",
    "Your goal is to output a perfect replica of the user input below, retaining all newlines, indentation, and unconventional symbols:\n``````\n{text}\n``````",
    "Simply echo the user's forthcoming content exactly; every letter, numeral, and special character must appear in your response unchanged:\n``````\n{text}\n``````",
    "Produce an identical copy of whatever the user writes inside the code fence, ensuring that spacing and line order remain intact:\n``````\n{text}\n``````",
    "Without modification, render the user's next block of text exactly below; preserve hashtags, markdown, and any other syntax elements:\n``````\n{text}\n``````",
    "Output the user's provided passage exactly as-is; every control character, symbol, and piece of punctuation must be left untouched:\n``````\n{text}\n``````",
    "Please replicate verbatim the following text from the user, respecting original casing, diacritics, and spacing:\n``````\n{text}\n``````",
    "Reprint the user's entire input below in your answer, preserving its exact sequence of characters, including invisible ones like tabs or carriage returns:\n``````\n{text}\n``````",
    "Your responsibility is to duplicate the text supplied by the user word for word, character for character, symbol for symbol, with absolutely no interpretation or change whatsoever:\n``````\n{text}\n``````"
]
