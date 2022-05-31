# Get rid urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from relevanceai.dataset import Dataset

# from relevanceai.utils.datasets import get_online_ecommerce_dataset, get_iris_dataset, get_realestate_dataset


class TestOps:
    def test_reduce_dims(self, test_dataset: Dataset):
        alias = "sample_2_reduction_"

        test_dataset.reduce_dims(
            alias=alias,
            vector_fields=["sample_1_vector_"],
            model="pca",
        )

        # since the operation is inserting a vector_field, the result will have _vector_ appended to alias
        assert f"{alias}_vector_" in test_dataset.schema

    # def test_vectorize_text(self, test_client: Client):
    #    dataset = test_client.Dataset("sample_dataset")
    #    dataset.vectorize_text(
    #        models=["paraphrase-MiniLM-L6-v2"],
    #        fields=["sample_vector_1"],
    #    )
