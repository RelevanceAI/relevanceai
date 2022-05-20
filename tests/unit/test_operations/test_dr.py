from relevanceai.dataset import Dataset


def test_reduce_dimensions(test_dataset: Dataset):
    model = "pca"
    alias = "pca"
    n_components = 3
    vector_field = "sample_1_vector_"
    test_dataset.reduce_dims(
        model=model,
        n_components=n_components,
        vector_fields=[vector_field],
        alias=alias,
    )

    dr_vector_name = f"{alias}_vector_"
    assert dr_vector_name in test_dataset.schema
