import pytest

import random

from dataclasses import dataclass

import uuid


@pytest.fixture(scope="session", autouse=True)
def test_dataclass_documents():
    @dataclass
    class Document:
        _id: str = uuid.uuid4().__str__()
        value1: float = random.random()
        value2: float = random.random()

    return [Document()] * 20
