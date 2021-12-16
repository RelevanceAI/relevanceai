from abc import abstractmethod
import numpy as np
import pandas as pd
from rank_bm25 import *
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS


class BM25:
    def __init__(self, corpus: list):
        self.df = pd.DataFrame(corpus, columns=["corpus"])
        self.df["corpus"] = self.df["corpus"].astype(str)
        self.df["tokens"] = self.df["corpus"].apply(word_tokenize)
        self.df["tokens"] = self.df["tokens"].apply(self.spl_chars_removal)
        self.df["cleaned_tokens"] = self.df["tokens"].apply(
            self.stopwords_removal_gensim_custom
        )
        self.corpus = self.df["cleaned_tokens"].tolist()
        self.bm25 = BM25Okapi(self.corpus)

    def spl_chars_removal(self, lst):
        lst1 = list()
        for element in lst:
            str = ""
            str = re.sub("[⁰-9a-zA-Z]”,” ", element)
            lst1.append(str)
        return lst1

    def stopwords_removal_gensim_custom(self, lst):
        all_stopwords_gensim = STOPWORDS
        lst1 = list()
        for str in lst:
            text_tokens = word_tokenize(str)
            tokens_without_sw = [
                word for word in text_tokens if not word in all_stopwords_gensim
            ]
            str_t = " ".join(tokens_without_sw)
            lst1.append(str_t)

        return lst1

    def search(self, query: str, k: int = 50):
        tokenised_query = query.split(" ")
        results = self.bm25.get_top_n(tokenised_query, self.corpus, n=k)
        return results

    """

    Insert a list of strings to create a BM25 object.
    The BM25 has a search method to find documents based on a string query
    The search method has a k parameter to find the top k documents, defaulting to 50

    Example
    ----
    corpus = ["I like dogs", "dogs are cool", "Felix is a cat"]
    bm25 = BM25(corpus)
    self.search("feline")
    self.search("feline", k=10)

    """
