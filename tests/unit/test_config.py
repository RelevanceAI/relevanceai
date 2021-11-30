import pytest
import random

# TODO: Fix status_code
#@pytest.mark.parametrize("output_format, expected_output_type", [("json", dict), ("content", bytes), ("status_code", int)])

@pytest.mark.parametrize("output_format, expected_output_type", [("json", dict), ("content", bytes)])
def test_output_format(test_client, test_sample_dataset, output_format, expected_output_type):
    test_client.output_format = output_format
    results = test_client.services.search.vector(
        test_sample_dataset,
        multivector_query=[
            {
                "vector": [random.randint(0, 1000) for _ in range(100)],
                "fields": ["sample_1_vector_"]
            }
        ]
    )
    assert isinstance(results, expected_output_type)