"""
Clean HTML
"""
import string
import re
import warnings

from typing import List, Optional
from collections import Counter
from html.parser import HTMLParser
from io import StringIO


class BaseTextProcessing:
    """Base text processing"""

    @staticmethod
    def remove_punctuation(text):
        return "".join([ch for ch in text if ch not in string.punctuation])

    @staticmethod
    def lower_text(text):
        return text.lower()

    @staticmethod
    def remove_html_tags(text):
        stripper = MLStripper()
        return stripper.clean(text)

    @staticmethod
    def remove_digits(text):
        return "".join([ch for ch in text if ch not in string.digits])

    @staticmethod
    def remove_stopwords(text: str, additional_stp_wrds: List[str] = None):
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        if additional_stp_wrds:
            stop_words = stop_words.union(set([w.lower() for w in additional_stp_wrds]))
        word_tokens = word_tokenize(text)
        return " ".join([w for w in word_tokens if w.lower() not in stop_words])

    @staticmethod
    def remove_url(text):
        return re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
            flags=re.MULTILINE,
        )

    @staticmethod
    def lemmatize(text: str):
        import nltk

        nltk.download("punkt")
        nltk.download("omw-1.4")
        nltk.download("wordnet")
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        # todo: find a better one => (NLTK changes less to le !!!)
        wordnet_lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        return " ".join([wordnet_lemmatizer.lemmatize(w) for w in word_tokens])

    @staticmethod
    def replace_words(text, replace_words: dict):
        for old, new in replace_words.items():
            if old in text:
                text = text.replace(old, new)
        return text

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
