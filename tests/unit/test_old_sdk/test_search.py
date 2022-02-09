import pytest

import random

from typing import Dict, List

from relevanceai import Client


def test_search_vector(test_client: Client, vector_dataset_id: str):
    results = test_client.services.search.vector(
        vector_dataset_id,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"],
            }
        ],
    )
    assert "results" in results


def test_suggestion(test_client: Client, vector_dataset_id: str):
    results = test_client.services.search.vector(
        vector_dataset_id,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"],
            }
        ],
    )
    suggestions = test_client.make_search_suggestion()
    assert "search" in suggestions, "Running a test"
