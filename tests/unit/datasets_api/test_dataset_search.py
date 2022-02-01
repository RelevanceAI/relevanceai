import pandas as pd
from relevanceai.http_client import Dataset, Client
from ...utils import generate_random_string, generate_random_vector

MULTIVECTOR_QUERY = [
    {"vector": generate_random_vector(N=100), "fields": ["sample_1_vector_"]}
]

CHUNK_MULTIVECTOR_QUERY = [
    {"vector": generate_random_vector(N=100), "fields": ["_chunk_.label_chunkvector_"]}
]


def test_vector_search(test_dataset_df: Dataset):
    results = test_dataset_df.vector_search(multivector_query=MULTIVECTOR_QUERY)
    assert True


def test_hybrid_search(test_dataset_df: Dataset):
    results = test_dataset_df.hybrid_search(
        multivector_query=MULTIVECTOR_QUERY, text="hey", fields=["sample_1_label"]
    )
    assert True


def test_chunk_search(test_dataset_df: Dataset):
    results = test_dataset_df.chunk_search(
        multivector_query=CHUNK_MULTIVECTOR_QUERY, chunk_field="_chunk_"
    )
    assert True


def test_multistep_chunk_search(test_dataset_df: Dataset):
    results = test_dataset_df.multistep_chunk_search(
        multivector_query=CHUNK_MULTIVECTOR_QUERY,
        first_step_multivector_query=MULTIVECTOR_QUERY,
        chunk_field="_chunk_",
    )
    assert True
