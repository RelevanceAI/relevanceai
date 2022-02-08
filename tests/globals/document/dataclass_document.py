import random

import uuid

from typing import NamedTuple

from dataclasses import dataclass


@dataclass
class DataclassDocument:
    _id: str = uuid.uuid4().__str__()
    value1: float = random.random()
    value2: float = random.random()


def dataclass_document(id: str) -> NamedTuple:
    return DataclassDocument(id=uuid.uuid4().__str__())
