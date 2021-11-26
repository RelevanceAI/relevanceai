import pytest
import random

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

# How to skip a test if it takes a long time
# Now this wiill only run if you ad a --flag
# @pytest.mark.slow
def test_hybrid_seach_suggestion(test_client, test_sample_vector_dataset):
    """Test our hybrid search endpoint
    """
    test_client.services.search.hybrid(
        test_sample_vector_dataset,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"]
            }
        ]
    )
    suggestions = test_client.make_search_suggestion()
    assert "search" in suggestions, "Running a test"
