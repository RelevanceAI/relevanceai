from relevanceai.http_client import Dataset


def test_reduce_dimensions(test_df: Dataset):

    OUTPUT_VECTOR_FIELD = "sample_1_vector_"
    ALIAS = "pca"

    test_df.reduce_dimensions(vector_fields=[OUTPUT_VECTOR_FIELD], alias="pca")

    vector_field_name = ".".join(["_dr_", ALIAS, OUTPUT_VECTOR_FIELD])
    assert vector_field_name in test_df.schema, "Did not reduce dimensions properly"


def test_auto_reduce_dimensions(test_df: Dataset):

    OUTPUT_VECTOR_FIELD = "sample_1_vector_"
    ALIAS = "pca-3"

    test_df.auto_reduce_dimensions(vector_fields=[OUTPUT_VECTOR_FIELD], alias=ALIAS)

    vector_field_name = ".".join(["_dr_", ALIAS, OUTPUT_VECTOR_FIELD])
    assert vector_field_name in test_df.schema, "Did not reduce dimensions properly"
