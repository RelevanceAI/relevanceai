from typing import Dict

TFHUB_MODELS: Dict[str, Dict[str, int]] = {
    "https://tfhub.dev/google/universal-sentence-encoder/4": {
        "vector_length": 512,
    },
}

SENTENCE_TRANSFORMER_MODELS: Dict[str, Dict[str, int]] = {
    "distilroberta-base-paraphrase-v1": {
        "vector_length": 768,
    },
    "xlm-r-distilroberta-base-paraphrase-v1": {
        "vector_length": 768,
    },
    "paraphrase-xlm-r-multilingual-v1": {
        "vector_length": 768,
    },
    "distilbert-base-nli-stsb-mean-tokens": {
        "vector_length": 768,
    },
    "bert-large-nli-stsb-mean-tokens": {
        "vector_length": 1024,
    },
    "roberta-base-nli-stsb-mean-tokens": {
        "vector_length": 768,
    },
    "roberta-large-nli-stsb-mean-tokens": {
        "vector_length": 1024,
    },
    "distilbert-base-nli-stsb-quora-ranking": {
        "vector_length": 768,
    },
    "distilbert-multilingual-nli-stsb-quora-ranking": {
        "vector_length": 768,
    },
    "distilroberta-base-msmarco-v1": {
        "vector_length": 768,
    },
    "distiluse-base-multilingual-cased-v2": {
        "vector_length": 512,
    },
    "xlm-r-bert-base-nli-stsb-mean-tokens": {
        "vector_length": 768,
    },
    "bert-base-wikipedia-sections-mean-tokens": {
        "vector_length": 768,
    },
    "LaBSE": {
        "vector_length": 768,
    },
    "average_word_embeddings_glove.6B.300d": {
        "vector_length": 300,
    },
    "average_word_embeddings_komninos": {
        "vector_length": 300,
    },
    "average_word_embeddings_levy_dependency": {
        "vector_length": 768,
    },
    "average_word_embeddings_glove.840B.300d": {
        "vector_length": 300,
    },
    "paraphrase-xlm-r-multilingual-v1": {
        "vector_length": 768,
    },
    "all-MiniLM-L6-v2": {
        "vector_length": 384,
    },
}
