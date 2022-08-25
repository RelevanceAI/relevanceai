import random

import pytest

from relevanceai import Client
from relevanceai.dataset import Dataset
from relevanceai.utils.datasets import (
    get_iris_dataset,
    get_online_ecommerce_dataset,
    get_palmer_penguins_dataset,
)
from relevanceai.utils.decorators.vectors import catch_errors


from tests.globals.constants import (
    SAMPLE_DATASET_DATASET_PREFIX,
    generate_random_label,
    generate_random_string,
    generate_random_vector,
    generate_random_integer,
)


MULTIVECTOR_QUERY = [
    {"vector": generate_random_vector(), "fields": ["sample_1_vector_"]}
]

CHUNK_MULTIVECTOR_QUERY = [
    {"vector": generate_random_vector(), "fields": ["_chunk_.label_chunkvector_"]}
]


class TestVectorizeOps:
    @pytest.mark.xfail(reason="vectorhub is not required")
    def test_vectorize(self, test_client: Client):
        dataset = test_client.Dataset(SAMPLE_DATASET_DATASET_PREFIX + "_ecom")

        documents = get_online_ecommerce_dataset()
        dataset.insert_documents(documents=documents)

        dataset.vectorize(create_feature_vector=True)

        assert True

    def test_numeric_vectorize(self, test_client: Client):
        dataset = test_client.Dataset(SAMPLE_DATASET_DATASET_PREFIX + "_iris")

        documents = get_iris_dataset()
        dataset.insert_documents(documents=documents)

        dataset.vectorize(create_feature_vector=True)

        assert "_dim4_feature_vector_" in dataset.schema

        dataset.vectorize(fields=["numeric"], create_feature_vector=True)

        assert "_dim4_feature_vector_" in dataset.schema

    @pytest.mark.xfail(reason="deprecated")
    def test_custom_vectorize(self, test_client: Client):
        dataset = test_client.Dataset(SAMPLE_DATASET_DATASET_PREFIX + "_penguins")

        documents = get_palmer_penguins_dataset()
        dataset.insert_documents(documents=documents)

        from relevanceai.operations.vector import Base2Vec

        class CustomTextEncoder(Base2Vec):
            __name__ = "CustomTextEncoder".lower()

            def __init__(self, *args, **kwargs):
                super().__init__()

                self.vector_length = 128
                self.model = lambda x: [
                    random.random() if "None" not in str(x) else random.random() + 10
                    for _ in range(self.vector_length)
                ]

            @catch_errors
            def encode(self, value):
                vector = self.model(value)
                return vector

        dataset.vectorize(
            encoders=dict(
                text=[
                    CustomTextEncoder(),
                ],
            ),
            create_feature_vector=True,
        )

        vectors = [
            "Comments_customtextencoder_vector_",
            "Species_customtextencoder_vector_",
            "Stage_customtextencoder_vector_",
        ]
        assert all(vector in dataset.schema for vector in vectors)

    # def test_vector_search(self, test_dataset: Dataset):
    #     test_dataset.vector_search(
    #         multivector_query=MULTIVECTOR_QUERY,
    #     )
    #     assert True

    # def test_hybrid_search(self, test_dataset: Dataset):
    #     test_dataset.hybrid_search(
    #         multivector_query=MULTIVECTOR_QUERY,
    #         text="hey",
    #         fields=["sample_1_label"],
    #     )
    #     assert True

    # def test_chunk_search(self, test_dataset: Dataset):
    #     test_dataset.chunk_search(
    #         multivector_query=CHUNK_MULTIVECTOR_QUERY,
    #         chunk_field="_chunk_",
    #     )
    #     assert True

    # def test_multistep_chunk_search(self, test_dataset: Dataset):
    #     test_dataset.multistep_chunk_search(
    #         multivector_query=CHUNK_MULTIVECTOR_QUERY,
    #         first_step_multivector_query=MULTIVECTOR_QUERY,
    #         chunk_field="_chunk_",
    #     )
    #     assert True
