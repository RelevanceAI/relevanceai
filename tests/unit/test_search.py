import pytest

def test_search_vector(test_client, test_sample_dataset):
    import random
    results = test_client.services.search.vector(
        test_sample_dataset,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"]
            }
        ]
    )
    print("----\n", results.keys(), "\n---")

    assert "results" in results

def test_suggestion(test_client, test_sample_dataset):
    import random

    results = test_client.services.search.vector(
        test_sample_dataset,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"]
            }
        ]
    )
    suggestions = test_client.make_search_suggestion()
    assert "search" in suggestions, "Running a test"
