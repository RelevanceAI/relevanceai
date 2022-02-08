import pytest

import random

import uuid

from tests.globals.utils import generate_random_vector


@pytest.fixture(scope="session")
def simple_document():
    return [
        {
            "_id": uuid.uuid4().__str__(),
            "value": random.randint(0, 1000),
            "sample_1_vector_": generate_random_vector(N=100),
        }
    ]
