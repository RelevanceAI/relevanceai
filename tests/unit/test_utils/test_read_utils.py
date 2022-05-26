import pytest

from relevanceai.utils import DocUtils


def test_is_field(sample_document):
    assert DocUtils().is_field("fehugieh", sample_document) is False
    key = list(sample_document.keys())[0]
    assert DocUtils().is_field("value", sample_document) is True
    assert DocUtils().is_field("blast32", sample_document) is True
    assert DocUtils().get_field("blast32", sample_document) == 21


def test_is_field_2(sample_document):
    assert DocUtils().is_field("blast32", sample_document) is True


@pytest.mark.skip(reason="Hell idk - need to think how to solve this problem")
def test_get_field(sample_document):
    assert (
        DocUtils().get_field("value.32", sample_document) == sample_document["value.32"]
    )
    assert DocUtils().get_field("value", sample_document) == sample_document["value"]


def test_get_field_across_documents(sample_document, sample_2_document):
    """Test to ensure you can get field across documents"""
    sample_docs = [sample_document] * 20 + [sample_2_document] * 100
    new_docs = DocUtils().get_field_across_documents(
        "value", sample_docs, missing_treatment="skip"
    )
    assert len(new_docs) == 20, "Not skipping"


def test_get_field_across_documents_for_skip_if_any_missing(
    sample_document, sample_2_document, combined_sample_document
):
    """Test to ensure you can get field across documents"""
    sample_docs = (
        [sample_document] * 20
        + [sample_2_document] * 100
        + [combined_sample_document] * 5
    )
    new_docs = DocUtils().get_fields_across_documents(
        ["value", "check_value"], sample_docs, missing_treatment="skip_if_any_missing"
    )
    assert len(new_docs) == 5, "Not skipping"


def test_subset_documents(combined_sample_document):
    sample_docs = [combined_sample_document] * 100
    subset_documents = DocUtils().subset_documents(
        ["value", "check_value"], sample_docs
    )
    for subset_doc in subset_documents:
        assert len(subset_doc) == 2
    assert len(subset_documents) == 100


def test_missing_treatment(combined_sample_document):
    assert -1 == DocUtils().get_field(
        "anonymous", combined_sample_document, missing_treatment=-1
    )


def test_inplace(sample_document):
    sample_document = sample_document

    new_sample_document = DocUtils().set_field(
        "blast32", sample_document, 22, inplace=False
    )

    assert sample_document["blast32"] == 21
    assert new_sample_document["blast32"] == 22
