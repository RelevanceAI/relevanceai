import random

from tests.globals.constants import generate_random_vector


def simple_document(_id: str):
    return {
        "_id": _id,
        "value": random.randint(0, 1000),
        "sample_1_vector_": generate_random_vector(),
    }
