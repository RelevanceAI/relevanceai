import pytest

from typing import Dict, List, NamedTuple

from relevanceai import Client


@pytest.fixture(scope="session")
def obj_dataset(
    test_client: Client,
    dataclass_documents: List[NamedTuple],
    test_dataset_id: str,
):
    test_client._insert_documents(test_dataset_id, dataclass_documents)

    yield test_dataset_id

    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def clustered_dataset(
    test_client: Client, vector_documents: List[Dict], test_dataset_id: str
):
    test_client._insert_documents(test_dataset_id, vector_documents)

    test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id=test_dataset_id,
        vector_fields=["sample_1_vector_"],
        k=10,
        alias="kmeans_10",
        overwrite=True,
    )

    yield test_dataset_id

    test_client.datasets.delete(test_dataset_id)
