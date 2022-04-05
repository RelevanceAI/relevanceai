from typing import Dict

from relevanceai.utils import make_id

from tests.globals.constants import (
    generate_random_label,
    generate_random_string,
    generate_random_vector,
    generate_random_integer,
)


def vector_document() -> Dict:
    document = {
        "sample_1_label": generate_random_label(),
        "sample_2_label": generate_random_label(),
        "sample_3_label": generate_random_label(),
        "sample_1_description": generate_random_string(),
        "sample_2_description": generate_random_string(),
        "sample_3_description": generate_random_string(),
        "sample_1_vector_": generate_random_vector(),
        "sample_2_vector_": generate_random_vector(),
        "sample_3_vector_": generate_random_vector(),
        "sample_1_value": generate_random_integer(),
        "sample_2_value": generate_random_integer(),
        "sample_3_value": generate_random_integer(),
        "_chunk_": [
            {
                "label": generate_random_label(),
                "label_chunkvector_": generate_random_vector(),
            }
        ],
    }
    document["_id"] = make_id(document)
    return document
