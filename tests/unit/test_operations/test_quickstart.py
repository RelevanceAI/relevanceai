"""Simple test to ensure the quickstarts work
"""
import time
from relevanceai import Client


def test_quickstart(test_client: Client):
    QUICKSTART_DATASET = "quickstart"
    documents = [
        {"_id": "1", "example_vector_": [0.1, 0.1, 0.1]},
        {"_id": "2", "example_vector_": [0.2, 0.2, 0.2]},
        {"_id": "3", "example_vector_": [0.3, 0.3, 0.3]},
        {"_id": "4", "example_vector_": [0.4, 0.4, 0.4]},
        {"_id": "5", "example_vector_": [0.5, 0.5, 0.5]},
    ]

    test_client._insert_documents(dataset_id=QUICKSTART_DATASET, documents=documents)
    time.sleep(2)
    results = test_client.services.search.vector(
        dataset_id=QUICKSTART_DATASET,
        multivector_query=[
            {"vector": [0.2, 0.2, 0.2], "fields": ["example_vector_"]},
        ],
        page_size=3,
    )
    assert len(results["results"]) > 0, "Not inserting properly"
    test_client.datasets.delete(QUICKSTART_DATASET)
