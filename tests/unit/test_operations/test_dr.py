from relevanceai.dataset import Dataset


def test_reduce_dimensions(test_dataset: Dataset):

    OUTPUT_VECTOR_FIELD = "sample_1_vector_"
    ALIAS = "pca"

    test_dataset.reduce_dimensions(vector_fields=[OUTPUT_VECTOR_FIELD], alias="pca")

    vector_field_name = ".".join(["_dr_", ALIAS, OUTPUT_VECTOR_FIELD])
    assert (
        vector_field_name in test_dataset.schema
    ), "Did not reduce dimensions properly"


def test_auto_reduce_dimensions(test_dataset: Dataset):

    OUTPUT_VECTOR_FIELD = "sample_1_vector_"
    ALIAS = "pca-3"

    test_dataset.auto_reduce_dimensions(
        vector_fields=[OUTPUT_VECTOR_FIELD], alias=ALIAS
    )

    vector_field_name = ".".join(["_dr_", ALIAS, OUTPUT_VECTOR_FIELD])
    assert (
        vector_field_name in test_dataset.schema
    ), "Did not reduce dimensions properly"
    # test_dataset.auto_cluster(
    #     alias="kmeans-2", vector_fields=["_dr_.pca-3.sample_1_vector_"]
    # )
    # assert True


# def test_clustering_on_dr(test_dataset: Dataset):
#     OUTPUT_VECTOR_FIELD = "sample_1_vector_"
#     ALIAS = "pca-3"
#     test_dataset.auto_reduce_dimensions(vector_fields=[OUTPUT_VECTOR_FIELD], alias=ALIAS)
#     test_dataset.auto_cluster(
#         alias="kmeans-2", vector_fields=["_dr_.pca-3.sample_1_vector_"]
#     )
#     assert "_cluster_._dr_.pca-3.sample_1_vector_.kmeans-2" in test_dataset.schema
