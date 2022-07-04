"""
    Test cleaning the operations
"""


def test_clean_dataset(test_dataset):
    # Simple smoke test to make sure that it runs
    test_dataset.clean_text(text_fields=["sample_1_label"])
    assert True
