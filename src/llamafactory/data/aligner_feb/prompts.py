#
# Created on Sun Mar 16 2025
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


SYSTEM_SYTLE_PROMPT = [
"""Your conversation style with users is:
``````
{style}
``````""",
"""Your dialogue manner with users is defined as:
``````
{style}
``````""",
"""Adopt the following conversation style when interacting with users:
``````
{style}
``````""",
"""The conversation style you should maintain with users is:
``````
{style}
``````""",
"""When conversing with users, use the following style:
``````
{style}
``````""",
"""Interact with users using the following conversation style:
``````
{style}
``````""",
"""Your chat style for engaging with users is:
``````
{style}
``````""",
"""The style of your conversation with users is as follows:
``````
{style}
``````""",
"""Please maintain this conversation style with users:
``````
{style}
``````""",
"""When interacting with users, your conversation style should be:
``````
{style}
``````""",
"""Your communication style with users is set as:
``````
{style}
``````""",
"""The way you interact with users should reflect the following style:
``````
{style}
``````""",
"""Your approach to conversing with users is defined by:
``````
{style}
``````""",
"""Use the following conversation style when communicating with users:
``````
{style}
``````""",
"""The style for your user interactions is:
``````
{style}
``````"""
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
    """You must repeat the user's input exactly as it is presented."""
]


USER_TTS_PROMPT = [
    """Please repeat the following user's input (which may contain some special symbols):

``````
{text}
``````""",
    """Your task is to echo the following user's input (which may include special symbols):

``````
{text}
``````""",
    """Repeat the following user's input exactly as provided, even if it contains special symbols:

``````
{text}
``````""",
    """Output the following user's input precisely as given (special symbols included):

``````
{text}
``````""",
    """Your assignment is to replicate the following user's input (note that it may include special symbols):

``````
{text}
``````""",
    """Please echo back the user's input exactly, including any special symbols:

``````
{text}
``````""",
    """Your job is to reproduce the following user's input verbatim (including special symbols):

``````
{text}
``````""",
    """Repeat exactly the user's input provided below (special symbols must be retained):

``````
{text}
``````""",
    """You are required to repeat the following user's input exactly as given (with special symbols intact):

``````
{text}
``````""",
    """Your task is to mirror the following user's input exactly (special symbols should remain unchanged):

``````
{text}
``````""",
    """Please duplicate the following user's input word-for-word (special symbols are permitted):

``````
{text}
``````""",
    """Repeat the following user's input exactly, including any special symbols:

``````
{text}
``````""",
    """Your assignment is to echo exactly the user's input shown below (special symbols included):

``````
{text}
``````""",
    """Your task is to output the following user's input precisely as provided (note: it may contain special symbols):

``````
{text}
``````""",
    """Repeat exactly the user's input provided below, ensuring that all special symbols are preserved:

``````
{text}
``````"""
]
