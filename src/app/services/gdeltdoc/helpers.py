# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                       │
# │  Project: MASX AI – Strategic Agentic AI System               │
# │  All rights reserved.                                         │
# └───────────────────────────────────────────────────────────────┘
#
# MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# The source code, documentation, workflows, designs, and naming (including "MASX AI")
# are protected by applicable copyright and trademark laws.
#
# Redistribution, modification, commercial use, or publication of any portion of this
# project without explicit written consent is strictly prohibited.
#
# This project is not open-source and is intended solely for internal, research,
# or demonstration use by the author.
#
# Contact: ab@masxai.com | MASXAI.com

from typing import Union

import json
from datetime import datetime

Date = Union[str, datetime]


def load_json(json_message, max_recursion_depth: int = 100, recursion_depth: int = 0):
    """
    tries to load a JSON formatted string and removes offending characters if present.

    :param json_message: The JSON string to load.
    :param max_recursion_depth: The maximum recursion depth allowed.
    :param recursion_depth: The current recursion depth.
    :return: The parsed JSON object.
    """
    try:
        if isinstance(json_message, bytes):
            json_message = json_message.decode()
        result = json.loads(json_message)

    except ValueError as e:
        if recursion_depth >= max_recursion_depth:
            raise ValueError("Max recursion depth is reached.")

        idx_to_replace = int(e.args[0].split(" ")[-1][:-1])
        json_message = (
            json_message[:idx_to_replace] + " " + json_message[idx_to_replace + 1 :]  # type: ignore
        )
        return load_json(json_message, max_recursion_depth, recursion_depth + 1)

    return result


def format_date(date: Date) -> str:
    """
    Takes a date as a string in YYYY-MM-DD format or as a datetime and returns it
    as a string formatted for the API (YYYYMMDDHHMMSS)
    """
    if type(date) == str:
        return f'{date.replace("-", "")}000000'
    elif type(date) == datetime:
        # it's a datetime
        return date.strftime("%Y%m%d%H%M%S")
    else:
        raise ValueError(f"Unsupported type for date: {type(date), {date}}")
