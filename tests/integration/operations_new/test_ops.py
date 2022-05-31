# Get rid urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from relevanceai import Client

# from relevanceai.utils.datasets import get_online_ecommerce_dataset, get_iris_dataset, get_realestate_dataset


class TestOps:
    def test_reduce_dims(self, test_client: Client):
        dataset = test_client.Dataset("sample_dataset")
        dataset.reduce_dims(
            alias="sample_2_reduction_",
            vector_fields=["sample_1_vector_"],
            model="pca",
        )

    # def test_vectorize_text(self, test_client: Client):
    #    dataset = test_client.Dataset("sample_dataset")
    #    dataset.vectorize_text(
    #        models=["paraphrase-MiniLM-L6-v2"],
    #        fields=["sample_vector_1"],
    #    )
