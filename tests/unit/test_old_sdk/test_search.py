import pytest

from relevanceai import Client

from tests.globals.constants import generate_random_vector


def test_search_vector(test_client: Client, vector_dataset_id: str):
    results = test_client.services.search.vector(
        vector_dataset_id,
        multivector_query=[
            {
                "vector": generate_random_vector(),
                "fields": ["sample_1_vector_"],
            }
        ],
    )
    assert "results" in results


def test_suggestion(test_client: Client, vector_dataset_id: str):
    test_client.services.search.vector(
        vector_dataset_id,
        multivector_query=[
            {
                "vector": generate_random_vector(),
                "fields": ["sample_1_vector_"],
            }
        ],
    )
    suggestions = test_client.make_search_suggestion()
    assert "search" in suggestions, "Running a test"
