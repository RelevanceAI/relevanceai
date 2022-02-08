import pytest

import uuid
import random

from tests.globals.utils import generate_random_vector


@pytest.fixture(scope="session", autouse=True)
def simple_doc():
    return [
        {
            "_id": uuid.uuid4().__str__(),
            "value": random.randint(0, 1000),
            "sample_1_vector_": generate_random_vector(N=100),
        }
    ]
