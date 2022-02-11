import random

from relevanceai.http_client import Dataset


def test_dataset_vectorize(test_df: Dataset):

    OUTPUT_VECTOR_FIELD = "sample_1_vector_"

    def encode(document):
        return [random.randint(0, 100) for _ in range(5)]

    def encode_document(document):
        document[OUTPUT_VECTOR_FIELD] = encode(document)

    def encode_documents(documents):
        for d in documents:
            encode_document(d)
        return documents

    test_df.vectorize("sample_1_label", encode_documents)
    assert OUTPUT_VECTOR_FIELD in test_df.schema, "Did not vectorize properly"
