"""
Test that analyzing text works out of the box
"""
import pytest
from relevanceai.dataset import Dataset


def test_analyze_text(test_dataset: Dataset):
    test_dataset.analyze_text(
        fields=["sample_1_label"],
        vector_fields=["sample_vector_"],
        vectorize=True,
        cluster=True,
        subcluster=True,
        extract_sentiment=True,
        extract_emotion=True,
    )
    assert "sample_vector_" in test_dataset.schema
