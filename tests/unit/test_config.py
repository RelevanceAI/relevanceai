import pytest

def test_output_format(test_client, test_sample_dataset):
    import random

    # JSON output
    test_client.output_format = "json"
    results = test_client.services.search.vector(
        test_sample_dataset,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"]
            }
        ]
    )
    assert isinstance(results, dict)

    # Content output
    test_client.output_format = "content"
    results = test_client.services.search.vector(
        test_sample_dataset,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"]
            }
        ]
    )
    assert isinstance(results, bytes)

    # Content status_code
    test_client.output_format = "status_code"
    results = test_client.services.search.vector(
        test_sample_dataset,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"]
            }
        ]
    )
    assert isinstance(results, int)