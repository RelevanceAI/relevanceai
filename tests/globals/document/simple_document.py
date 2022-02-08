import pytest

import random

from tests.globals.utils import generate_random_vector


@pytest.fixture(scope="session")
def simple_document(id: str):
    return [
        {
            "_id": id,
            "value": random.randint(0, 1000),
            "sample_1_vector_": generate_random_vector(N=100),
        }
    ]
