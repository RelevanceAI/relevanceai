import pytest

import uuid

from tests.globals.utils import (
    generate_random_label,
    generate_random_string,
    generate_random_vector,
    generate_random_integer,
)


@pytest.fixture(scope="session", autouse=True)
def sample_vector_documents():
    def _sample_vector_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1_label": generate_random_label(),
            "sample_2_label": generate_random_label(),
            "sample_3_label": generate_random_label(),
            "sample_1_description": generate_random_string(),
            "sample_2_description": generate_random_string(),
            "sample_3_description": generate_random_string(),
            "sample_1_vector_": generate_random_vector(N=100),
            "sample_2_vector_": generate_random_vector(N=100),
            "sample_3_vector_": generate_random_vector(N=100),
            "sample_1_value": generate_random_integer(N=100),
            "sample_2_value": generate_random_integer(N=100),
            "sample_3_value": generate_random_integer(N=100),
            "_chunk_": [
                {
                    "label": generate_random_label(),
                    "label_chunkvector_": generate_random_vector(100),
                }
            ],
        }

    N = 100
    return [_sample_vector_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]
