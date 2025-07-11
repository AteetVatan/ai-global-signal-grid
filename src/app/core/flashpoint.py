from typing import List
from pydantic import BaseModel, Field, field_validator, RootModel
#from pydantic_core import RootModel

class FlashpointItem(BaseModel):
    title: str = Field(..., min_length=1, description="Title of the flashpoint event")
    description: str = Field(..., min_length=1, description="Brief description of the event")
    entities: List[str] = Field(..., description="List of named entities involved")

    @field_validator("title", "description", mode="before")
    @classmethod
    def strip_and_validate_non_empty(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Must be a non-empty, non-whitespace string")
        return v.strip()
    
    @field_validator("entities", mode="before")
    def ensure_non_empty_entities(cls, value):
        if not value or not isinstance(value, list) or any(not isinstance(e, str) for e in value):
            raise ValueError("Entities must be a list of non-empty strings")
        return value




class FlashpointDataset(RootModel[List[FlashpointItem]]):
    root: List[FlashpointItem] = Field(default_factory=list) # to avoid FlashpointDataset(root=[])
    
    @classmethod
    def from_raw(cls, raw):
        if not raw:
            return cls()
        #Fix: Ensure we're working with list of dicts
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
