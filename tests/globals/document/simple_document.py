import random

from relevanceai.utils import make_id

from tests.globals.constants import generate_random_vector


def simple_document():
    document = {
        "value": random.randint(0, 1000),
        "sample_1_vector_": generate_random_vector(),
        "random_category": random.choice(["red", "blue", "yellow"]),
    }
    document["_id"] = make_id(document)
    return document
