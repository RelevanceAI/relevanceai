"""
Clean HTML
"""
import string
import warnings

from typing import List, Optional
from collections import Counter
from html.parser import HTMLParser
from io import StringIO


class BaseTextProcessing:
    """Base text processing"""

    @staticmethod
    def normalize_text(
        txt: str,
        lower: bool = True,
        remove_digit: bool = True,
        remove_punct: bool = True,
    ) -> str:
        """
        * Lower-casing
        * Digit removal
        * Punctuation removal
        """
        if lower:
            txt = txt.lower()
        if remove_digit:
            txt = "".join([ch for ch in txt if ch not in string.digits])
        if remove_punct:
            txt = "".join([ch for ch in txt if ch not in string.punctuation])
        return txt

    @staticmethod
    def get_word_frequency(
        str_list: List[str],
        remove_stop_words: bool = True,
        additional_stop_words: Optional[List[str]] = None,
        language="english",
    ) -> List:
        """Returns a sorted word frequency in Python"""
        additional_stop_words = (
            [] if additional_stop_words is None else additional_stop_words
        )
        try:
            import nltk

            nltk.download("stopwords")
            from nltk.corpus import stopwords
        except ModuleNotFoundError:
            warnings.warn("You are missing NLTK, please run `pip install nltk`")

        if remove_stop_words:
            stpw = stopwords.words(language)
            stpw += additional_stop_words
        else:
            stpw = []
        word_counter = Counter(
            [w.lower() for s in str_list for w in s.split() if w not in stpw]
        )
        return sorted(word_counter.items(), key=lambda item: (-item[1], item[0]))


class MLStripper(HTMLParser):
    """Remove HTML from the code and retrieves data."""

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return (
            self.text.getvalue()
            .replace("\r", "")
            .replace("\n", "")
            .replace("\t", " ")
            .strip()
        )

    def clean(self, text):
        self.reset()
        self.handle_data(text)
        return self.get_data()
