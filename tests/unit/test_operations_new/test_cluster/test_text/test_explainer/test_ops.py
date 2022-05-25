"""Simple test for testing cluster ops
"""
import pytest
from relevanceai.dataset import Dataset


@pytest.mark.skip(reason="not implemented properly")
def test_cluster_ops_explain_closest(test_dataset: Dataset):
    # Get the cluster ops object
    # Using the cluster ops object, highlight the closest ones
    # highlight the closest ones

    VECTOR_FIELDS = ["sample_1_vector_"]
    ALIAS = "explainer"

    cluster_ops = test_dataset.cluster(vector_fields=VECTOR_FIELDS, alias=ALIAS)

    from relevanceai.operations_new.cluster.ops import ClusterOps as NewClusterOps

    new_cluster_ops = NewClusterOps(
        dataset_id=test_dataset.dataset_id, vector_fields=VECTOR_FIELDS, alias=ALIAS
    )
    results = new_cluster_ops.explain_text_clusters(
        text_field="sample_1_label", n_closest=5
    )
    assert results
