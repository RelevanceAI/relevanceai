import random

from typing import NamedTuple

from dataclasses import dataclass

from relevanceai.package_utils.make_id import _make_id


@dataclass
class DataclassDocument:
    _id: str = None
    value1: float = random.random()
    value2: float = random.random()


def dataclass_document() -> NamedTuple:
    document = DataclassDocument()
    document._id = _make_id(document)
    return document
