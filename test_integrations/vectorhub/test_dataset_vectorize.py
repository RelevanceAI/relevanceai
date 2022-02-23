from relevanceai.http_client import Dataset


def test_dataset_vectorize(test_dataset: Dataset):
    results = test_dataset.vectorize(image_fields=["image_url"], text_fields=["data"])
    # This means the results are good yay!
    assert "image_url_clip_vector_" in test_dataset.schema
    assert "data_use_vector_" in test_dataset.schema
