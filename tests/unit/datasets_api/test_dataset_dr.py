from relevanceai.http_client import Dataset


def test_reduce_dimensiosn(test_dataset_df: Dataset):

    OUTPUT_VECTOR_FIELD = "sample_1_vector_"
    ALIAS = "pca"

    test_dataset_df.reduce_dimensions(vector_fields=[OUTPUT_VECTOR_FIELD], alias="pca")

    vector_field_name = ".".join(["_dr_", ALIAS, OUTPUT_VECTOR_FIELD])
    assert (
        vector_field_name in test_dataset_df.schema
    ), "Did not reduce dimensions properly"
