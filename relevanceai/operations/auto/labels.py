# -*- coding: utf-8 -*-
"""
Pandas like dataset API
"""
import warnings
from relevanceai.constants.warning import Warning
import itertools

from itertools import chain
from collections import Counter

from typing import Callable, Dict, List, Optional

from tqdm.auto import tqdm

from relevanceai.dataset.write.write import Write

from relevanceai.operations.vector.local_nearest_neighbours import NearestNeighbours
from relevanceai.operations.preprocessing.text.base_text_processing import MLStripper
from relevanceai.operations.cluster.constants import NEAREST_NEIGHBOURS

from relevanceai.utils.decorators.analytics import track
from relevanceai.utils.logger import FileLogger
from relevanceai.utils.decorators.version import beta

# TODO: Separate out operations into different files - cluster/search/dr


class Labels(Write):
    @track
    def label_vector(
        self,
        vector,
        alias: str,
        label_dataset_id: str,
        label_vector_field: str,
        label_fields: list,
        number_of_labels: int = 1,
        similarity_metric: NEAREST_NEIGHBOURS = "cosine",
        score_field: str = "_search_score",
        **kwargs,
    ):
        """
        Label a dataset based on a model.

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems.

        .. note::
            **New in v0.32.0**

        Parameters
        -------------

        vector_fields: list
            The list of vector field
        label_dataset_id: str
            The dataset to label with
        alias: str
            The alias of the labels (for example - "ranking_labels")
        label_dataset_id: str
            The dataset to use fo rlabelling
        label_vector_field: str
            The vector field of the label dataset
        label_fields: list
            The label field of the dataset to use
        number_of_labels: int
            The numebr of labels to get
        similarity_metric: str
            The similarity metric to adopt
        score_field: str
            The field to use for scoring

        Example
        -------------

        .. code-block::


            from relevanceai import Client
            from relevanceai.ops.clusterops.cluster import ClusterOps
            from relevanceai.ops.clusterops.kmeans_clusterer import KMeansModel

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            result = df.label_vector(
                [...],
                label_vector_field="sample_1_vector_",
                alias="alias_sample",
                label_dataset_id=label_dataset_id_id,
                label_fields=["sample_1_label"],
                number_of_labels=1,
            )

        """
        # Download documents in the label dataset
        label_documents: list = self._get_all_documents(
            label_dataset_id, select_fields=[label_vector_field] + label_fields
        )

        # Build a index
        labels = self._get_nearest_labels(
            label_documents=label_documents,
            vector=vector,
            label_vector_field=label_vector_field,
            similarity_metric=similarity_metric,
            number_of_labels=number_of_labels,
            score_field=score_field,
            label_fields=label_fields,
        )

        # Store things according to
        # {"_label_": {"field": {"alias": [{"label": 3, "similarity_score": 0.4}]}
        return self.store_labels_in_document(labels, alias)

    @track
    def store_labels_in_document(self, labels: list, alias: str):
        if isinstance(labels, dict) and "label" in labels:
            return {"_label_": {alias: labels["label"]}}
        return {"_label_": {alias: labels}}

    def _get_nearest_labels(
        self,
        label_documents: List[Dict],
        vector: List[float],
        label_vector_field: str,
        similarity_metric: NEAREST_NEIGHBOURS,
        number_of_labels: int,
        label_fields: List[str],
        score_field="_label_score",
    ):
        nearest_neighbors: List[Dict] = NearestNeighbours.get_nearest_neighbours(
            label_documents,
            vector,
            label_vector_field,
            similarity_metric,
            score_field=score_field,
        )[:number_of_labels]
        labels: List[Dict] = self.subset_documents(
            [score_field] + label_fields, nearest_neighbors
        )
        # Preprocess labels for easier frontend access
        new_labels = {}
        for lf in label_fields:
            new_labels[lf] = [
                {"label": l.get(lf), score_field: l.get(score_field)} for l in labels
            ]
        return new_labels

    @track
    def label_document(
        self,
        document: dict,
        vector_field: str,
        vector: List[float],
        alias: str,
        label_dataset_id: str,
        label_vector_field: str,
        label_fields: List[str],
        number_of_labels: int = 1,
        similarity_metric="cosine",
        score_field: str = "_search_score",
    ):
        """
        Label a dataset based on a model.

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems.

        .. note::
            **New in v0.32.0**

        Parameters
        -------------

        document: dict
            A document to label
        vector_field: str
            The list of vector field
        label_dataset_id: str
            The dataset to label with
        alias: str
            The alias of the labels (for example - "ranking_labels")
        label_dataset_id: str
            The dataset to use fo rlabelling
        label_vector_field: str
            The vector field of the label dataset
        label_fields: list
            The label field of the dataset to use
        number_of_labels: int
            The numebr of labels to get
        similarity_metric: str
            The similarity metric to adopt
        score_field: str
            The field to use for scoring

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset_id")

            results = df.label_document(
                document={...},
                vector_field="sample_1_vector_",
                alias="example",
                label_dataset_id=label_dataset_id_id,
                label_fields=["sample_1_label"],
                label_vector_field="sample_1_vector_",
                filters=[
                    {
                        "field": "sample_1_label",
                        "filter_type": "exists",
                        "condition": ">=",
                        "condition_value": " ",
                    },
                ],
            )
        """
        vector = self.get_field(vector_field, document)
        labels = self.label_vector(
            vector_field=vector_field,
            vector=vector,
            alias=alias,
            label_dataset_id=label_dataset_id,
            label_vector_field=label_vector_field,
            label_fields=label_fields,
            number_of_labels=number_of_labels,
            score_field=score_field,
            similarity_metric=similarity_metric,
        )
        document.update(self.store_labels_in_document(labels, alias))
        return document

    @track
    def label_from_dataset(
        self,
        vector_field: str,
        alias: str,
        label_dataset_id: str,
        label_vector_field: str,
        label_fields: List[str],
        number_of_labels: int = 1,
        filters: Optional[list] = None,
        similarity_metric="cosine",
        score_field: str = "_search_score",
    ):
        """
        Label a dataset based on a model.

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems.

        .. note::
            **New in v0.32.0**

        Parameters
        -------------

        vector_field: str
            The vector field to match with
        alias: str
            The alias of the labels (for example - "ranking_labels")
        label_dataset_id: str
            The dataset to use fo rlabelling
        label_vector_field: str
            The vector field of the label dataset
        label_fields: list
            The label field of the dataset to use
        filters: list
            The filters to apply to label
        number_of_labels: int
            The numebr of labels to get
        similarity_metric: str
            The similarity metric to adopt
        score_field: str
            The field to use for scoring

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset_id")

            results = df.label(
                vector_field="sample_1_vector_",
                alias="example",
                label_dataset_id=label_dataset_id,
                label_fields=["sample_1_label"],
                label_vector_field="sample_1_vector_",
                filters=[
                    {
                        "field": "sample_1_label",
                        "filter_type": "exists",
                        "condition": ">=",
                        "condition_value": " ",
                    },
                ],
            )

        """
        filters = [] if filters is None else filters

        # Download documents in the label dataset
        filters += [
            {
                "field": label_field,
                "filter_type": "exists",
                "condition": ">=",
                "condition_value": " ",
            }
            for label_field in label_fields
        ] + [
            {
                "field": label_vector_field,
                "filter_type": "exists",
                "condition": ">=",
                "condition_value": " ",
            },
        ]
        label_documents: list = self._get_all_documents(
            label_dataset_id,
            select_fields=[label_vector_field] + label_fields,
            filters=filters,
        )

        def label_and_store(d: dict):
            labels = self._get_nearest_labels(
                label_documents=label_documents,
                vector=self.get_field(vector_field, d),
                label_vector_field=label_vector_field,
                similarity_metric=similarity_metric,
                number_of_labels=number_of_labels,
                score_field=score_field,
                label_fields=label_fields,
            )
            d.update(self.store_labels_in_document(labels, alias))
            return d

        def bulk_label_documents(documents):
            [label_and_store(d) for d in documents]
            return documents

        return self.bulk_apply(
            bulk_label_documents,
            filters=[
                {
                    "field": vector_field,
                    "filter_type": "exists",
                    "condition": ">=",
                    "condition_value": " ",
                },
            ],
        )

    @track
    def label_from_list(
        self,
        vector_field: str,
        model: Callable,
        label_list: list,
        similarity_metric="cosine",
        number_of_labels: int = 1,
        score_field: str = "_search_score",
        alias: Optional[str] = None,
    ):
        """Label from a given list.

        Parameters
        ------------

        vector_field: str
            The vector field to label in the original dataset
        model: Callable
            This will take a list of strings and then encode them
        label_list: List
            A list of labels to accept
        similarity_metric: str
            The similarity metric to accept
        number_of_labels: int
            The number of labels to accept
        score_field: str
            What to call the scoring of the labels
        alias: str
            The alias of the labels

        Example
        --------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")

            # Get a model to help us encode
            from vectorhub.encoders.text.tfhub import USE2Vec
            enc = USE2Vec()

            # Use that model to help with encoding
            label_list = ["dog", "cat"]

            df = client.Dataset("_github_repo_vectorai")

            df.label_from_list("documentation_vector_", enc.bulk_encode, label_list, alias="pets")

        """
        if alias is None:
            warnings.warn(
                "No alias is detected for labelling. Default to 'default' as the alias."
            )
            alias = "default"
        print("Encoding labels...")
        label_vectors = []
        for c in self.chunk(label_list, chunksize=20):
            with FileLogger(verbose=True):
                label_vectors.extend(model(c))

        if len(label_vectors) == 0:
            raise ValueError("Failed to encode.")

        # we need this to mock label documents - these values are not important
        # and can be changed :)
        LABEL_VECTOR_FIELD = "label_vector_"
        LABEL_FIELD = "label"

        label_documents = [
            {LABEL_VECTOR_FIELD: label_vectors[i], LABEL_FIELD: label}
            for i, label in enumerate(label_list)
        ]

        return self._bulk_label_dataset(
            label_documents=label_documents,
            vector_field=vector_field,
            label_vector_field=LABEL_VECTOR_FIELD,
            similarity_metric=similarity_metric,
            number_of_labels=number_of_labels,
            score_field=score_field,
            label_fields=[LABEL_FIELD],
            alias=alias,
        )

    def _bulk_label_dataset(
        self,
        label_documents,
        vector_field,
        label_vector_field,
        similarity_metric,
        number_of_labels,
        score_field,
        label_fields,
        alias,
    ):
        def label_and_store(d: dict):
            labels = self._get_nearest_labels(
                label_documents=label_documents,
                vector=self.get_field(vector_field, d),
                label_vector_field=label_vector_field,
                similarity_metric=similarity_metric,
                number_of_labels=number_of_labels,
                score_field=score_field,
                label_fields=label_fields,
            )
            d.update(self.store_labels_in_document(labels, alias))
            return d

        def bulk_label_documents(documents):
            [label_and_store(d) for d in documents]
            return documents

        print("Labelling dataset...")
        return self.bulk_apply(
            bulk_label_documents,
            filters=[
                {
                    "field": vector_field,
                    "filter_type": "exists",
                    "condition": ">=",
                    "condition_value": " ",
                },
            ],
            select_fields=[vector_field],
        )

    def _set_up_nltk(
        self,
        stopwords_dict: str = "english",
        additional_stopwords: Optional[list] = None,
    ):
        """Additional stopwords to include"""
        import nltk
        from nltk.corpus import stopwords

        additional_stopwords = (
            [] if additional_stopwords is None else additional_stopwords
        )

        self._is_set_up = True
        nltk.download("stopwords")
        nltk.download("punkt")
        self.eng_stopwords = stopwords.words(stopwords_dict)
        self.eng_stopwords.extend(additional_stopwords)
        self.eng_stopwords = set(self.eng_stopwords)

    def clean_html(self, html):
        """Cleans HTML from text"""
        s = MLStripper()
        if html is None:
            return ""
        s.feed(html)
        return s.get_data()

    def get_word_count(self, text_fields: List[str]):
        """
        Create labels from a given text field.

        Parameters
        ------------

        text_fields: list
            List of text fields


        Example
        ------------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            df.get_word_count()

        """
        import nltk
        from nltk.corpus import stopwords

        raise NotImplementedError

    def generate_text_list_from_documents(
        self,
        documents: Optional[list] = None,
        text_fields: Optional[list] = None,
        clean_html: bool = False,
    ):
        """
        Generate a list of text from documents to feed into the counter
        model.
        Parameters
        -------------
        documents: list
            A list of documents
        fields: list
            A list of fields
        clean_html: bool
            If True, also cleans the text in a given text document to remove HTML. Will be slower
            if processing on a large document
        """
        documents = [] if documents is None else documents
        text_fields = [] if text_fields is None else text_fields

        text = self.get_fields_across_documents(
            text_fields, documents, missing_treatment="ignore"
        )
        return list(itertools.chain.from_iterable(text))

    def generate_text_list(
        self,
        filters: Optional[list] = None,
        batch_size: int = 20,
        text_fields: Optional[list] = None,
        cursor: str = None,
    ):
        filters = [] if filters is None else filters
        text_fields = [] if text_fields is None else text_fields

        filters += [
            {
                "field": tf,
                "filter_type": "exists",
                "condition": "==",
                "condition_value": " ",
            }
            for tf in text_fields
        ]
        documents = self.get_documents(
            batch_size=batch_size,
            select_fields=text_fields,
            filters=filters,
            cursor=cursor,
        )
        return self.generate_text_list_from_documents(
            documents, text_fields=text_fields
        )

    def get_ngrams(
        self,
        text,
        n: int = 2,
        stopwords_dict: str = "english",
        additional_stopwords: Optional[list] = None,
        min_word_length: int = 2,
        preprocess_hooks: Optional[list] = None,
    ):
        additional_stopwords = (
            [] if additional_stopwords is None else additional_stopwords
        )
        preprocess_hooks = [] if preprocess_hooks is None else preprocess_hooks

        try:
            return self._get_ngrams(
                text=text,
                n=n,
                additional_stopwords=additional_stopwords,
                min_word_length=min_word_length,
                preprocess_hooks=preprocess_hooks,
            )
        except:
            # Specify that this shouldn't necessarily error out.
            self.set_up(
                stopwords_dict=stopwords_dict, additional_stopwords=additional_stopwords
            )
            return self._get_ngrams(
                text=text,
                n=n,
                min_word_length=min_word_length,
                preprocess_hooks=preprocess_hooks,
            )

    def _get_ngrams(
        self,
        text,
        n: int = 2,
        additional_stopwords: Optional[list] = None,
        min_word_length: int = 2,
        preprocess_hooks: Optional[list] = None,
    ):
        """Get the bigrams"""
        from nltk import word_tokenize
        from nltk.util import ngrams

        additional_stopwords = (
            [] if additional_stopwords is None else additional_stopwords
        )
        preprocess_hooks = [] if preprocess_hooks is None else preprocess_hooks

        if additional_stopwords:
            [self.eng_stopwords.add(s) for s in additional_stopwords]

        n_grams = []
        for line in text:
            for p_hook in preprocess_hooks:
                line = p_hook(line)
            token = word_tokenize(line)
            n_grams.append(list(ngrams(token, n)))

        def length_longer_than_min_word_length(x):
            return len(x.strip()) >= min_word_length

        def is_not_stopword(x):
            return x.strip() not in self.eng_stopwords

        def is_clean(text_list):
            return all(
                length_longer_than_min_word_length(x) and is_not_stopword(x)
                for x in text_list
            )

        counter = Counter([" ".join(x) for x in chain(*n_grams) if is_clean(x)])
        return counter
        # return dict(counter.most_common(most_common))

    @track
    @beta
    def keyphrases(
        self,
        text_fields: list,
        algorithm: str = "rake",
        n: int = 2,
        most_common: int = 10,
        filters: Optional[list] = None,
        additional_stopwords: Optional[list] = None,
        min_word_length: int = 2,
        batch_size: int = 1000,
        document_limit: int = None,
        preprocess_hooks: Optional[List[callable]] = None,
        verbose: bool = True,
    ) -> list:
        """
        Returns the most common phrase in the following format:

        .. code-block::

            [('heavily draping faux fur', 16.0),
            ('printed sweatshirt fabric made', 14.333333333333334),
            ('high paper bag waist', 14.25),
            ('ribbed organic cotton jersey', 13.803030303030303),
            ('soft sweatshirt fabric', 9.0),
            ('open back pocket', 8.5),
            ('layered tulle skirt', 8.166666666666666),
            ('soft brushed inside', 8.0),
            ('discreet side pockets', 7.5),
            ('cotton blend', 5.363636363636363)]

        Parameters
        ------------

        text_fields: list
            A list of text fields
        algorithm: str
            The algorithm to use. Must be one of `nltk` or `rake`.
        n: int
            if algorithm is `nltk`, this will set the number of words. If `rake`, then it
            will do nothing.
        most_common: int
            How many to return
        filters: list
            A list of filters to supply
        additional_stopwords: list
            A list of additional stopwords to supply
        min_word_length: int
            The minimum word length to apply to clean. This can be helpful if there are common
            acronyms that you want to exclude.
        batch_size: int
            Batch size is the number of documents to retrieve in a chunk
        document_limit: int
            The maximum number of documents in a dataset
        preprocess_hooks: List[Callable]
            A list of process hooks to clean text before they count as a word

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            # Returns the top keywords in a text field
            ds.keyphrases(text_fields=["sample"])


            # Create an e-commerce dataset

            from relevanceai.package_utils.datasets import get_dummy_ecommerce_dataset
            docs = get_dummy_ecommerce_dataset()
            ds = client.Dataset("ecommerce-example")
            ds.upsert_documents(docs)
            ds.keyphrases(text_fields=text_fields, algorithm="nltk", n=3)
            def remove_apostrophe(string):
                return string.replace("'s", "")
            ds.keyphrases(text_fields=text_fields, algorithm="nltk", n=3, preprocess_hooks=[remove_apostrophe])
            ds.keyphrases(text_fields=text_fields, algorithm="nltk", n=3, additional_stopwords=["Men", "Women"])

        """
        self._check_keyphrase_algorithm_requirements(algorithm)
        filters = [] if filters is None else filters
        additional_stopwords = (
            [] if additional_stopwords is None else additional_stopwords
        )
        preprocess_hooks = [] if preprocess_hooks is None else preprocess_hooks

        counter: Counter = Counter()
        if not hasattr(self, "_is_set_up"):
            if verbose:
                print("setting up NLTK...")
            self._set_up_nltk()

        # Mock a dummy documents so I can loop immediately without weird logic
        documents: dict = {"documents": [[]], "cursor": None}
        if verbose:
            print("Updating word count...")
        while len(documents["documents"]) > 0 and (
            document_limit is None or sum(counter.values()) < document_limit
        ):
            # TODO: make this into a progress bar instead
            documents = self.get_documents(
                filters=filters,
                cursor=documents["cursor"],
                batch_size=batch_size,
                select_fields=text_fields,
                include_cursor=True,
            )
            string = self.generate_text_list_from_documents(
                documents=documents["documents"],
                text_fields=text_fields,
            )

            if algorithm == "nltk":
                ngram_counter = self._get_ngrams(
                    string,
                    n=n,
                    additional_stopwords=additional_stopwords,
                    min_word_length=min_word_length,
                    preprocess_hooks=preprocess_hooks,
                )
            elif algorithm == "rake":
                ngram_counter = self._get_rake_keyphrases(
                    string,
                )
            counter.update(ngram_counter)
        return counter.most_common(most_common)

    def _check_keyphrase_algorithm_requirements(self, algorithm: str):
        if algorithm == "rake":
            try:
                import rake_nltk
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Run `pip install rake-nltk`.")
        elif algorithm == "nltk":
            try:
                import nltk
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Run `pip install nltk`")

    def _get_rake_keyphrases(self, string, **kw):
        from rake_nltk import Rake

        r = Rake(**kw)
        r.extract_keywords_from_sentences(string)
        results = r.get_ranked_phrases_with_scores()
        return {v: k for k, v in results}

    def cluster_keyphrases(
        self,
        vector_fields: List[str],
        text_fields: List[str],
        cluster_alias: str,
        cluster_field: str = "_cluster_",
        num_clusters: int = 100,
        most_common: int = 10,
        preprocess_hooks: Optional[List[callable]] = None,
        algorithm: str = "rake",
        n: int = 2,
    ):
        """
        Simple implementation of the cluster word cloud.

        Parameters
        ------------
        vector_fields: list
            The list of vector fields
        text_fields: list
            The list of text fields
        cluster_alias: str
            The alias of the cluster
        cluster_field: str
            The cluster field to try things on
        num_clusters: int
            The number of clusters
        preprocess_hooks: list
            The preprocess hooks
        algorithm: str
            The algorithm to use
        n: int
            The number of words

        """
        preprocess_hooks = [] if preprocess_hooks is None else preprocess_hooks

        vector_fields_str = ".".join(sorted(vector_fields))
        field = f"{cluster_field}.{vector_fields_str}.{cluster_alias}"
        all_clusters = self.facets([field], page_size=num_clusters)
        cluster_counters = {}
        if "results" in all_clusters:
            all_clusters = all_clusters["results"]
        # TODO: Switch to multiprocessing
        for c in tqdm(all_clusters[field]):
            cluster_value = c[field]
            top_words = self.keyphrases(
                text_fields=text_fields,
                n=n,
                filters=[
                    {
                        "field": field,
                        "filter_type": "contains",
                        "condition": "==",
                        "condition_value": cluster_value,
                    }
                ],
                most_common=most_common,
                preprocess_hooks=preprocess_hooks,
                algorithm=algorithm,
            )
            cluster_counters[cluster_value] = top_words
        return cluster_counters

    # TODO: Add keyphrases to auto cluster
    # def auto_cluster_keyphrases(
    #     vector_fields: List[str],
    #     text_fields: List[str],
    #     cluster_alias: str,
    #     deployable_id: str,
    #     n: int = 2,
    #     cluster_field: str = "_cluster_",
    #     num_clusters: int = 100,
    #     preprocess_hooks: Optional[List[callable]] = None,
    # ):
    #     """
    #     # TODO:
    #     """
    #     pass

    def _add_cluster_word_cloud_to_config(self, data, cluster_value, top_words):
        # TODO: Add this to wordcloud deployable
        # hacky way I implemented to add top words to config
        data["configuration"]["cluster-labels"][cluster_value] = ", ".join(
            [k for k in top_words if k != "machine learning"]
        )
        data["configuration"]["cluster-descriptions"][cluster_value] = str(top_words)

    @track
    def label_from_common_words(
        self,
        text_field: str,
        model: Callable = None,
        most_common: int = 1000,
        n_gram: int = 1,
        temp_vector_field: str = "_label_vector_",
        labels_fn="labels.txt",
        stopwords: Optional[list] = None,
        algorithm: str = "nltk",
    ):
        """
        Label by the most popular keywords.

        Algorithm:

        - Get top X keywords or bigram for a text field
        - Default X to 1000 or something scaled towards number of documents
        - Vectorize those into keywords
        - Label every document with those top keywords

        .. note::
            **New in v1.1.0**

        Parameters
        ------------

        text_fields: str
            The field to label
        model: Callable
            The function or callable to turn text into a vector.
        most_common: int
            How many of the most common worsd do you want to use as labels
        n_gram: int
            How many word co-occurrences do you want to consider
        temp_vector_field: str
            The temporary vector field name
        labels_fn: str
            The filename for labels to be saved in.
        stopwords: list
            A list of stopwords
        algorithm: str
            The algorithm to use. Must be one of `nltk` or `rake`.

        Example
        --------

        .. code-block::

            import random
            from relevanceai import Client
            from relevanceai.package_utils.datasets import mock_documents
            from relevanceai.package_utils.logger import FileLogger

            client = Client()
            ds = client.Dataset("sample")
            documents = mock_documents()
            ds.insert_documents(documents)

            def encode():
                return [random.randint(0, 100) for _ in range(5)]

            ds.label_from_common_words(
                text_field="sample_1_label",
                model=encode,
                most_common=10,
                n_gram=1
            )

        """
        stopwords = [] if stopwords is None else stopwords

        labels = self.keyphrases(
            text_fields=[text_field],
            n=n_gram,
            most_common=most_common,
            additional_stopwords=stopwords,
            algorithm=algorithm,
        )

        with open(labels_fn, "w") as f:
            f.write(str(labels))

        print(f"Saved labels to {labels_fn}")
        # Add support if already encoded
        def encode(doc):
            with FileLogger():
                doc[temp_vector_field] = model(self.get_field(text_field, doc))
            return doc

        self.apply(encode, select_fields=[text_field])
        # create_labels_from_text_field
        return self.label_from_list(
            vector_field=temp_vector_field,
            model=model,
            label_list=[x[0] for x in labels],
        )
