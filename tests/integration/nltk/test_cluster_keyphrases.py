"""Testing the cluster keyphrases
"""
import os
from relevanceai.client import Client
from relevanceai.operations.labels.labels import LabelOps


def test_cluster_keyphrases(test_client: Client, test_dataset_id: str):
    os.system("pip install nltk")
    ds = test_client.Dataset(test_dataset_id)
    VECTOR_FIELDS = ["sampel_1_vector_"]
    TEXT_FIELDS = ["category_1"]
    ALIAS = "sample_label"

    label_ops = LabelOps.from_dataset(ds, alias=ALIAS, vector_fields=VECTOR_FIELDS)

    keyphrases = label_ops.cluster_keyphrases(
        vector_fields=VECTOR_FIELDS,
        text_fields=TEXT_FIELDS,
        cluster_alias=ALIAS,
        most_common=2,
        num_clusters=3,
        algorithm="nltk",  # can be one of nltk or nltk-rake
        n=2,  # number of co-occurring words to use
        preprocess_hooks=[],  # A list of functions to parse in a string to clean
    )
