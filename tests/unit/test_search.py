import pytest
import random


def test_search_vector(test_client, test_sample_vector_dataset):
    results = test_client.services.search.vector(
        test_sample_vector_dataset,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"],
            }
        ],
    )
    assert "results" in results


def test_suggestion(test_client, test_sample_vector_dataset):
    results = test_client.services.search.vector(
        test_sample_vector_dataset,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"],
            }
        ],
    )
    suggestions = test_client.make_search_suggestion()
    assert "search" in suggestions, "Running a test"
