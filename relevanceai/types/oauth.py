from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class OAuth(BaseModel):
    auth_url: str
    scopes: Optional[List[str]] = None


from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class Token(BaseModel):
    permission_types: List[str]
    scopes: List[str]
    token_id: str
    metadata: Dict[str, str]
    insert_date_: datetime


class Account(BaseModel):
    _id: str
    project: str
    provider: str
    account_id: str
    provider_user_id: str
    insert_date_: datetime
    update_date_: datetime
    tokens: List[Token]
    metadata: Dict[str, str] = Field(default_factory=dict)
    label: str

    def __repr__(self) -> str:
        return f"Integration(provider='{self.provider}, email='{self.label}'"


class ActiveIntegrations(BaseModel):
    results: List[Account]

    def __getattr__(self, provider: str) -> Optional[Account]:
        if provider in ["google"]:  # Add supported providers
            return next((r for r in self.results if r.provider == provider), None)
        raise AttributeError(
            f"'{self.__class__.__name__}' has no attribute '{provider}'"
        )
