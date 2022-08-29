import time
from relevanceai.dataset import Dataset

from sklearn.cluster import KMeans


class TestOperation:
    def test_subcluster(self, test_dataset: Dataset):
        model = KMeans(n_clusters=4)

        vector_field = "sample_1_vector_"
        alias = "cluster_test_1"
        test_dataset.cluster(
            model=model,
            vector_fields=[vector_field],
            alias=alias,
            include_cluster_report=False,
        )
        time.sleep(1)
        assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema

        parent_field = f"_cluster_.{vector_field}.{alias}"
        # vector_field = "sample_2_vector_"
        alias = "subcluster_test_1"
        test_dataset.subcluster(
            model=model,
            alias=alias,
            vector_fields=[vector_field],
            parent_field=parent_field,
            filters=[
                {
                    "field": vector_field,
                    "filter_type": "exists",
                    "condition": "==",
                    "condition_value": "",
                }
            ],
            min_parent_cluster_size=4,
        )
        time.sleep(1)
        assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema
