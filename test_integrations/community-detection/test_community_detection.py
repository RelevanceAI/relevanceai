from relevanceai.interfaces import Dataset


def test_community_detection(test_dataset: Dataset):
    # test text field first
    text_field = "data"
    test_dataset.community_detection(field=text_field)
    assert f"_cluster_.{text_field}.community-detection" in test_dataset.schema

    # vectorize a field to test that community detection works on vectors
    test_dataset.vectorize(image_fields=["image_url"])
    vector_field = "image_url_clip_vector_"
    test_dataset.community_detection(field=vector_field)
    assert f"_cluster_.{vector_field}.community-detection" in test_dataset.schema
