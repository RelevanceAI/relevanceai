import pytest
import random

from typing import Dict, List

from relevanceai import Client


@pytest.mark.skip(reason="failed")
def test_output_format(test_client: Client, sample_dataset_id: List[Dict]):
    output_format = "content"
    expected_output_type = bytes
    test_client.output_format = output_format
    results = test_client.services.search.vector(
        sample_dataset_id,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"],
            }
        ],
    )
    assert isinstance(results, expected_output_type)


@pytest.mark.skip(reason="failed")
def test_output_format_json(test_client: Client, sample_dataset_id: List[Dict]):
    output_format = "json"
    expected_output_type = dict
    test_client.output_format = output_format
    results = test_client.services.search.vector(
        sample_dataset_id,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"],
            }
        ],
    )
    assert isinstance(results, expected_output_type)
