"""
Test cloning and sending a dataset. Simple tests to check functionality.
In truth, should be handled API side.
"""
import time
from relevanceai import Client


def test_sending_dataset(test_client: Client, test_sample_dataset):

    DATASET_ID = "_sample_dataset_"
    NEW_DATASET_ID = DATASET_ID + "-2"
    docs = [{"_id": "10", "value": 10}, {"_id": "1000", "value": 30}]
    test_client.insert_documents(DATASET_ID, docs)
    test_client.send_dataset(
        dataset_id=DATASET_ID,
        receiver_project=test_client.project,
        receiver_api_key=test_client.api_key,
    )
    test_client.delete_dataset(DATASET_ID)
    assert True


def test_cloning_dataset(test_client: Client, test_sample_dataset):
    # Create a really samll dataset
    DATASET_ID = "_sample_dataset_"
    NEW_DATASET_ID = DATASET_ID + "-2"
    docs = [{"_id": "10", "value": 10}, {"_id": "1000", "value": 30}]
    test_client.insert_documents(DATASET_ID, docs)
    test_client.clone_dataset(
        source_dataset_id=DATASET_ID, new_dataset_id=NEW_DATASET_ID
    )
    time.sleep(10)
    results = test_client.delete_dataset(NEW_DATASET_ID)
    assert True
