"""Simple test to ensure the quickstarts work
"""
import time

def test_quickstart(test_client):
    docs = [
        {"_id": "1", "example_vector_": [0.1, 0.1, 0.1]},
        {"_id": "2", "example_vector_": [0.2, 0.2, 0.2]},
        {"_id": "3", "example_vector_": [0.3, 0.3, 0.3]},
        {"_id": "4", "example_vector_": [0.4, 0.4, 0.4]},
        {"_id": "5", "example_vector_": [0.5, 0.5, 0.5]},
    ]

    test_client.insert_documents(dataset_id="quickstart", docs=docs)
    time.sleep(2)
    results = test_client.services.search.vector(
        dataset_id="quickstart", 
        multivector_query=[
            {"vector": [0.2, 0.2, 0.2], "fields": ["example_vector_"]},
        ],
        page_size=3
    )
    assert len(results['results']) > 0, "Not inserting properly"
