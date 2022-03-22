"""
Test the clustering workflow from getting the documents, clustering and then inserting the relevant centroids
"""

import pytest

from relevanceai import Client
from relevanceai.interfaces import Dataset

from typing import List

from relevanceai.workflows.cluster_ops.ops import ClusterOps
from relevanceai.workflows.cluster_ops.base import CentroidClusterBase


@pytest.mark.skip(reason="no longer supported")
@pytest.mark.parametrize(
    "vector_fields", [["sample_1_vector_"], ["sample_2_vector_", "sample_1_vector_"]]
)
def test_cluster_integration_one_liner(
    test_client: Client, vector_dataset_id: str, vector_fields: List
):
    """Smoke Test for the entire clustering workflow."""
    # Retrieve a previous dataset
    VECTOR_FIELDS = vector_fields
    test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id=vector_dataset_id,
        vector_fields=VECTOR_FIELDS,
        overwrite=True,
        alias="sample_cluster",
    )
    assert True
