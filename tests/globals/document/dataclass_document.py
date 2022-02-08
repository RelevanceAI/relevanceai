import pytest

import random

import uuid

from typing import NamedTuple

from dataclasses import dataclass


@dataclass
class DataclassDocument:
    _id: str = uuid.uuid4().__str__()
    value1: float = random.random()
    value2: float = random.random()


@pytest.fixture(scope="session")
def dataclass_document() -> NamedTuple:
    return DataclassDocument()
