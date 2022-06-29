"""
Test that analyzing text works out of the box
"""
import pytest
from relevanceai.dataset import Dataset
from typing import List


@pytest.mark.parametrize(
    ["vectorize_models", "subcluster_model", "filters"],
    [
        (
            ["princeton-nlp/sup-simcse-roberta-large"],
            "all-mpnet-base-v2",
            None,
        )
    ],
    ids=["single_model"],
)
def test_analyze_text(
    test_dataset: Dataset,
    vectorize_models: List,
    subcluster_model: str,
    filters: List[dict],
):
    test_vector_name = "sample_vector_"
    test_field = "sample_1_label"

    test_dataset.analyze_text(
        fields=[test_field],
        vector_fields=[test_vector_name],
        vectorize_models=vectorize_models,
        vectorize=True,
        cluster=True,
        subcluster=True,
        subcluster_model=subcluster_model,
        extract_sentiment=True,
        extract_emotion=True,
        count=True,
        filters=filters,
    )
    assert test_vector_name in test_dataset.schema
