from relevanceai.dataset import Dataset


def test_reduce_dimensions(test_dataset: Dataset):
    model = "pca"
    alias = "pca"
    dims = 3
    vector_field = "sample_1_vector_"
    test_dataset.reduce_dims(
        model=model,
        dims=dims,
        vector_fields=[vector_field],
        alias=alias,
    )

    dr_vector_name = f"_dr_.{alias}.{vector_field}"
    assert dr_vector_name in test_dataset.schema
