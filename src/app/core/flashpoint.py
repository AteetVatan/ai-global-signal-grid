# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# │  Project: MASX AI – Strategic Agentic AI System              │
# │  All rights reserved.                                        │
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

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, RootModel
from ..core.querystate import QueryState

# from pydantic_core import RootModel


class FlashpointItem(BaseModel):
    
    title: str = Field(..., min_length=1, description="Title of the flashpoint event")
    description: str = Field(
        ..., min_length=1, description="Brief description of the event"
    )
    entities: List[str] = Field(..., description="List of named entities involved")
    domains: Optional[List[str]] = Field(
        None, description="List of domains the event belongs to"
    )
    queries: Optional[List[QueryState]] = Field(None, description="List of queries")

    @field_validator("title", "description", mode="before")
    @classmethod
    def strip_and_validate_non_empty(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Must be a non-empty, non-whitespace string")
        return v.strip()

    @field_validator("entities", mode="before")
    def ensure_non_empty_entities(cls, value):
        if (
            not value
            or not isinstance(value, list)
            or any(not isinstance(e, str) for e in value)
        ):
            raise ValueError("Entities must be a list of non-empty strings")
        return value

    @field_validator("domains", mode="before")
    @classmethod
    def validate_domains(cls, value):
        if value is None:
            return value  # Skip validation
        if not isinstance(value, list) or any(
            not isinstance(d, str) or not d.strip() for d in value
        ):
            raise ValueError("Domains must be a list of non-empty strings")
        return [d.strip() for d in value]


class FlashpointDataset(RootModel[List[FlashpointItem]]):
    root: List[FlashpointItem] = Field(
        default_factory=list
    )  # to avoid FlashpointDataset(root=[])

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            return cls()
        # Fix: Ensure we're working with list of dicts
        if isinstance(raw[0], FlashpointItem):
            raw = [fp.model_dump() for fp in raw]
        return cls.model_validate(raw)

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return len(self.root)

    def to_list(self) -> List[dict]:
        return [item.model_dump() for item in self.root]

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_list(), ensure_ascii=False, indent=2)

    def append(self, item: FlashpointItem):
        if not isinstance(item, FlashpointItem):
            raise TypeError("Only FlashpointItem instances can be added.")
        self.root.append(item)

    def extend(self, items: List[FlashpointItem]):
        for item in items:
            self.append(item)
