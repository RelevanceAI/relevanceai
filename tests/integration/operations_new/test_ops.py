# Get rid of urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import pytest

from relevanceai.dataset import Dataset


class TestOps:
    def test_reduce_dims(self, test_dataset: Dataset):
        alias = "reduce_dims_test"

        test_dataset.reduce_dims(
            alias=alias,
            vector_fields=["sample_1_vector_"],
            model="pca",
        )
        # since the operation is inserting a vector_field, the result will have _vector_ appended to alias
        assert f"{alias}_vector_" in test_dataset.schema

    def test_vectorize_text(self, test_dataset: Dataset):
        model = "paraphrase-MiniLM-L6-v2"
        field = "vectorize_text"

        test_dataset.vectorize_text(
            fields=[field],
            models=[model],
        )
        assert f"{field}_{model}_vector_" in test_dataset.schema

    @pytest.mark.skip(reason="Takes forever to complete")
    def test_extract_sentiment(self, test_dataset: Dataset):
        test_dataset.extract_sentiment(
            text_fields=["sample_1_vector_"],
            model_name="siebert/sentiment-roberta-large-english",
            highlight=False,
        )
        assert False

    @pytest.mark.skip(reason="Got AssertionError: Torch not compiled with CUDA enabled")
    def test_apply_transformers_pipeline(self, test_dataset: Dataset):
        from transformers import pipeline

        pipeline_ = pipeline(
            "automatic-speech-recognition",
            model="facebook/wav2vec2-base-960h",
            device=0,
        )

        test_dataset.apply_transformers_pipeline(
            text_fields=["sample_1_description"],
            pipeline=pipeline_,
        )
        assert False
