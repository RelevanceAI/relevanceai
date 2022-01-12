from relevanceai.http_client import Dataset


def test_datasets_api(test_dataset_df: Dataset):
    """Testing the datasets API
    Simple smoke tests for now until we are happy with functionality :)
    """
    test_dataset_df.info()
    test_dataset_df.describe()
    test_dataset_df.head()
    assert True
