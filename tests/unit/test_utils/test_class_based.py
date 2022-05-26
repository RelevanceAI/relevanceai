import pytest

from doc_utils import Document
from doc_utils import DocumentList


def test_document(sample_document):
    document = Document(sample_document)

    label_val = sample_document["value"]

    target_value = document["value"]
    assert label_val == target_value

    document["value"] = 3
    assert document["value"] == 3


def test_nested_document(sample_nested_document):
    document = Document(sample_nested_document)

    label_val = sample_nested_document["value1"]["value2"]["value3"]

    target_value = document["value1.value2.value3"]
    assert label_val == target_value

    document["value1.value2.value3"] = 3
    assert document["value1.value2.value3"] == 3


def test_documents(sample_documents):
    documents = DocumentList(sample_documents)
    assert documents == documents


def test_document_json(sample_documents):
    documents = DocumentList(sample_documents)
    json_docs = documents.json()
    assert all(
        str(sample_document) == str(json_doc)
        for sample_document, json_doc in zip(sample_documents, json_docs)
    )


def test_document_methods(sample_nested_document):
    document = Document(sample_nested_document)

    keys = document.keys()
    values = document.values()
    items = document.items()

    assert keys
    assert values
    assert items
