import random

from relevanceai.package_utils.make_id import _make_id

from tests.globals.constants import generate_random_vector


def simple_document():
    document = {
        "value": random.randint(0, 1000),
        "sample_1_vector_": generate_random_vector(),
    }
    document["_id"] = _make_id(document)
    return document
