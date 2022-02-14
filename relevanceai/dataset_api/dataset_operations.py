# -*- coding: utf-8 -*-
"""
Pandas like dataset API
"""
import warnings
import itertools
from collections import Counter
from typing import Dict, List, Optional, Callable
from tqdm.auto import tqdm
from relevanceai.analytics_funcs import track
from relevanceai.dataset_api.dataset_write import Write
from relevanceai.dataset_api.dataset_series import Series
from relevanceai.data_tools.base_text_processing import MLStripper
from relevanceai.vector_tools.nearest_neighbours import (
    NearestNeighbours,
    NEAREST_NEIGHBOURS,
)

from relevanceai.logger import FileLogger


class Operations(Write):
    @track
    def vectorize(self, field: str, model):
        """
        Vectorizes a Particular field (text) of the dataset

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems. We recommend using
            the bulk_apply function or the apply function to provide the intended output
            for now.

        Parameters
        ------------
        field : str
            The text field to select
        model
            a Type deep learning model that vectorizes text

        Example
            -------
        .. code-block::

            from relevanceai import Client
            from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec

            model = SentenceTransformer2Vec("all-mpnet-base-v2 ")

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            text_field = "text_field"
            df.vectorize(text_field, model)
        """
        return Series(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            firebase_uid=self.firebase_uid,
            field=field,
        ).vectorize(model)

    @track
    def cluster(self, model, alias, vector_fields, **kwargs):
        """
        Performs KMeans Clustering on over a vector field within the dataset.

        .. warning::
            Deprecated in v0.33 in favour of df.auto_cluster.

        Parameters
        ----------
        model : Class
            The clustering model to use
        vector_fields : str
            The vector fields over which to cluster

        Example
        -------
        .. code-block::

            from relevanceai import Client
            from relevanceai.clusterer import ClusterOps
            from relevanceai.clusterer.kmeans_clusterer import KMeansModel

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            vector_field = "vector_field_"
            n_clusters = 10

            model = KMeansModel(k=n_clusters)

            df.cluster(model=model, alias=f"kmeans-{n_clusters}", vector_fields=[vector_field])
        """
        from relevanceai.clusterer import ClusterOps

        clusterer = ClusterOps(
            model=model,
            alias=alias,
            api_key=self.api_key,
            project=self.project,
            firebase_uid=self.firebase_uid,
        )
        clusterer.fit_predict_update(dataset=self, vector_fields=vector_fields)
        return clusterer

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
            from relevanceai.clusterer import ClusterOps
            from relevanceai.clusterer.kmeans_clusterer import KMeansModel

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
        filters: list = [],
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
        self, stopwords_dict: str = "english", additional_stopwords: list = []
    ):
        """Additional stopwords to include"""
        import nltk
        from nltk.corpus import stopwords

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
        self, documents: list = [], text_fields: list = [], clean_html: bool = False
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
        text = self.get_fields_across_documents(
            text_fields, documents, missing_treatment="ignore"
        )
        return list(itertools.chain.from_iterable(text))

    def generate_text_list(
        self,
        filters: list = [],
        batch_size: int = 20,
        text_fields: list = [],
        cursor: str = None,
    ):
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
        additional_stopwords: list = [],
        min_word_length: int = 2,
        preprocess_hooks: list = [],
    ):
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
        additional_stopwords=[],
        min_word_length: int = 2,
        preprocess_hooks: list = [],
    ):
        """Get the bigrams"""

        from nltk import word_tokenize
        from nltk.util import ngrams
        from itertools import chain

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
    def get_wordcloud(
        self,
        text_fields: list,
        n: int = 2,
        most_common: int = 10,
        filters: list = [],
        additional_stopwords: list = [],
        min_word_length: int = 2,
        batch_size: int = 1000,
        document_limit: int = None,
        preprocess_hooks: List[callable] = [],
    ) -> list:
        """
        wordcloud return object looks like:

        .. code-block::

            {'python': 232}

        Parameters
        ------------

        text_fields: list
            A list of text fields

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()

        """
        counter: Counter = Counter()
        if not hasattr(self, "_is_set_up"):
            print("setting up NLTK...")
            self._set_up_nltk()

        # Mock a dummy documents so I can loop immediately without weird logic
        documents: dict = {"documents": [[]], "cursor": None}
        print("Updating word count...")
        while len(documents["documents"]) > 0 and (
            document_limit is None or sum(counter.values()) < document_limit
        ):
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

            ngram_counter = self._get_ngrams(
                string,
                n=n,
                additional_stopwords=additional_stopwords,
                min_word_length=min_word_length,
                preprocess_hooks=preprocess_hooks,
            )
            counter.update(ngram_counter)
        return counter.most_common(most_common)

    def cluster_word_cloud(
        self,
        vector_fields: List[str],
        text_fields: List[str],
        cluster_alias: str,
        n: int = 2,
        cluster_field: str = "_cluster_",
        num_clusters: int = 100,
        preprocess_hooks: List[callable] = [],
    ):
        """
        Simple implementation of the cluster word cloud
        """
        vector_fields_str = ".".join(sorted(vector_fields))
        field = f"{cluster_field}.{vector_fields_str}.{cluster_alias}"
        all_clusters = self.facets([field], page_size=num_clusters)
        most_common = 10
        cluster_counters = {}
        for c in tqdm(all_clusters[field]):
            cluster_value = c[field]
            top_words = self.get_wordcloud(
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
            )
            cluster_counters[c] = top_words
        return cluster_counters

    def _add_cluster_word_cloud_to_config(self, data, cluster_value, top_words):
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
        stopwords: list = [],
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

        Example
        --------

        .. code-block::

            import random
            from relevanceai import Client
            from relevanceai.datasets import mock_documents
            from relevanceai.logger import FileLogger

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
        labels = self.get_wordcloud(
            text_fields=[text_field],
            n=n_gram,
            most_common=most_common,
            additional_stopwords=stopwords,
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

    @track
    def vector_search(
        self,
        multivector_query: List,
        positive_document_ids: dict = {},
        negative_document_ids: dict = {},
        vector_operation="sum",
        approximation_depth=0,
        sum_fields=True,
        page_size=20,
        page=1,
        similarity_metric="cosine",
        facets=[],
        filters=[],
        min_score=0,
        select_fields=[],
        include_vector=False,
        include_count=True,
        asc=False,
        keep_search_history=False,
        hundred_scale=False,
        search_history_id=None,
        query: str = None,
    ):
        """
        Allows you to leverage vector similarity search to create a semantic search engine. Powerful features of VecDB vector search:

        1. Multivector search that allows you to search with multiple vectors and give each vector a different weight.
        e.g. Search with a product image vector and text description vector to find the most similar products by what it looks like and what its described to do.
        You can also give weightings of each vector field towards the search, e.g. image_vector_ weights 100%, whilst description_vector_ 50% \n
            An example of a simple multivector query:

            >>> [
            >>>     {"vector": [0.12, 0.23, 0.34], "fields": ["name_vector_"], "alias":"text"},
            >>>     {"vector": [0.45, 0.56, 0.67], "fields": ["image_vector_"], "alias":"image"},
            >>> ]

            An example of a weighted multivector query:

            >>> [
            >>>     {"vector": [0.12, 0.23, 0.34], "fields": {"name_vector_":0.6}, "alias":"text"},
            >>>     {"vector": [0.45, 0.56, 0.67], "fields": {"image_vector_"0.4}, "alias":"image"},
            >>> ]

            An example of a weighted multivector query with multiple fields for each vector:

            >>> [
            >>>     {"vector": [0.12, 0.23, 0.34], "fields": {"name_vector_":0.6, "description_vector_":0.3}, "alias":"text"},
            >>>     {"vector": [0.45, 0.56, 0.67], "fields": {"image_vector_"0.4}, "alias":"image"},
            >>> ]

        2. Utilise faceted search with vector search. For information on how to apply facets/filters check out datasets.documents.get_where \n
        3. Sum Fields option to adjust whether you want multiple vectors to be combined in the scoring or compared in the scoring. e.g. image_vector_ + text_vector_ or image_vector_ vs text_vector_. \n
            When sum_fields=True:

            - Multi-vector search allows you to obtain search scores by taking the sum of these scores.
            - TextSearchScore + ImageSearchScore = SearchScore
            - We then rank by the new SearchScore, so for searching 1000 documents there will be 1000 search scores and results

            When sum_fields=False:

            - Multi vector search but not summing the score, instead including it in the comparison!
            - TextSearchScore = SearchScore1
            - ImagSearchScore = SearchScore2
            - We then rank by the 2 new SearchScore, so for searching 1000 documents there should be 2000 search scores and results.

        4. Personalization with positive and negative document ids.

            - For more information about the positive and negative document ids to personalize check out services.recommend.vector

        For more even more advanced configuration and customisation of vector search, reach out to us at dev@relevance.ai and learn about our new advanced_vector_search.

        Parameters
        ----------

        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        positive_document_ids : dict
            Positive document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        negative_document_ids: dict
            Negative document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        approximation_depth: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
        vector_operation: string
            Aggregation for the vectors when using positive and negative document IDs, choose from ['mean', 'sum', 'min', 'max', 'divide', 'mulitple']
        sum_fields : bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        facets: list
            Fields to include in the facets, if [] then all
        filters: list
            Query for filtering the search results
        min_score: int
            Minimum score for similarity metric
        select_fields: list
            Fields to include in the search results, empty array/list means all fields.
        include_vector: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        asc: bool
            Whether to sort results by ascending or descending order
        keep_search_history: bool
            Whether to store the history into VecDB. This will increase the storage costs over time.
        hundred_scale: bool
            Whether to scale up the metric by 100
        search_history_id: string
            Search history ID, only used for storing search histories.
        query: string
            What to store as the query name in the dashboard

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            results = df.vector_search(multivector_query=MULTIVECTOR_QUERY)

        """

        return self.services.search.vector(
            dataset_id=self.dataset_id,
            multivector_query=multivector_query,
            positive_document_ids=positive_document_ids,
            negative_document_ids=negative_document_ids,
            vector_operation=vector_operation,
            approximation_depth=approximation_depth,
            sum_fields=sum_fields,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            facets=facets,
            filters=filters,
            min_score=min_score,
            select_fields=select_fields,
            include_vector=include_vector,
            include_count=include_count,
            asc=asc,
            keep_search_history=keep_search_history,
            hundred_scale=hundred_scale,
            search_history_id=search_history_id,
            query=query,
        )

    @track
    def hybrid_search(
        self,
        multivector_query: List,
        text: str,
        fields: list,
        edit_distance: int = -1,
        ignore_spaces: bool = True,
        traditional_weight: float = 0.075,
        page_size: int = 20,
        page=1,
        similarity_metric="cosine",
        facets=[],
        filters=[],
        min_score=0,
        select_fields=[],
        include_vector=False,
        include_count=True,
        asc=False,
        keep_search_history=False,
        hundred_scale=False,
        search_history_id=None,
    ):
        """
        Combine the best of both traditional keyword faceted search with semantic vector search to create the best search possible. \n

        For information on how to use vector search check out services.search.vector. \n

        For information on how to use traditional keyword faceted search check out services.search.traditional.

        Parameters
        ----------
        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        text : string
            Text Search Query (not encoded as vector)
        fields : list
            Text fields to search against
        positive_document_ids : dict
            Positive document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        negative_document_ids: dict
            Negative document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        approximation_depth: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
        vector_operation: string
            Aggregation for the vectors when using positive and negative document IDs, choose from ['mean', 'sum', 'min', 'max', 'divide', 'mulitple']
        sum_fields : bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        facets: list
            Fields to include in the facets, if [] then all
        filters: list
            Query for filtering the search results
        min_score: float
            Minimum score for similarity metric
        select_fields: list
            Fields to include in the search results, empty array/list means all fields.
        include_vector: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        asc: bool
            Whether to sort results by ascending or descending order
        keep_search_history: bool
            Whether to store the history into VecDB. This will increase the storage costs over time.
        hundred_scale: bool
            Whether to scale up the metric by 100
        search_history_id: string
            Search history ID, only used for storing search histories.
        edit_distance: int
            This refers to the amount of letters it takes to reach from 1 string to another string. e.g. band vs bant is a 1 word edit distance. Use -1 if you would like this to be automated.
        ignore_spaces: bool
            Whether to consider cases when there is a space in the word. E.g. Go Pro vs GoPro.
        traditional_weight: int
            Multiplier of traditional search score. A value of 0.025~0.075 is the ideal range


        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            MULTIVECTOR_QUERY = [{"vector": [0, 1, 2], "fields": ["sample_vector_"]}]
            results = df.vector_search(multivector_query=MULTIVECTOR_QUERY)

        """
        return self.services.search.hybrid(
            dataset_id=self.dataset_id,
            multivector_query=multivector_query,
            text=text,
            fields=fields,
            edit_distance=edit_distance,
            ignore_spaces=ignore_spaces,
            traditional_weight=traditional_weight,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            facets=facets,
            filters=filters,
            min_score=min_score,
            select_fields=select_fields,
            include_vector=include_vector,
            include_count=include_count,
            asc=asc,
            keep_search_history=keep_search_history,
            hundred_scale=hundred_scale,
            search_history_id=search_history_id,
        )

    @track
    def chunk_search(
        self,
        multivector_query,
        chunk_field,
        chunk_scoring="max",
        chunk_page_size: int = 3,
        chunk_page: int = 1,
        approximation_depth: int = 0,
        sum_fields: bool = True,
        page_size: int = 20,
        page: int = 1,
        similarity_metric: str = "cosine",
        facets: list = [],
        filters: list = [],
        min_score: int = None,
        include_vector: bool = False,
        include_count: bool = True,
        asc: bool = False,
        keep_search_history: bool = False,
        hundred_scale: bool = False,
        query: str = None,
    ):
        """
        Chunks are data that has been divided into different units. e.g. A paragraph is made of many sentence chunks, a sentence is made of many word chunks, an image frame in a video. By searching through chunks you can pinpoint more specifically where a match is occuring. When creating a chunk in your document use the suffix "chunk" and "chunkvector". An example of a document with chunks:

        >>> {
        >>>     "_id" : "123",
        >>>     "title" : "Lorem Ipsum Article",
        >>>     "description" : "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged.",
        >>>     "description_vector_" : [1.1, 1.2, 1.3],
        >>>     "description_sentence_chunk_" : [
        >>>         {"sentence_id" : 0, "sentence_chunkvector_" : [0.1, 0.2, 0.3], "sentence" : "Lorem Ipsum is simply dummy text of the printing and typesetting industry."},
        >>>         {"sentence_id" : 1, "sentence_chunkvector_" : [0.4, 0.5, 0.6], "sentence" : "Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book."},
        >>>         {"sentence_id" : 2, "sentence_chunkvector_" : [0.7, 0.8, 0.9], "sentence" : "It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged."},
        >>>     ]
        >>> }

        For combining chunk search with other search check out services.search.advanced_chunk.

        Parameters
        ----------

        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        chunk_field : string
            Field where the array of chunked documents are.
        chunk_scoring: string
            Scoring method for determining for ranking between document chunks.
        chunk_page_size: int
            Size of each page of chunk results
        chunk_page: int
            Page of the chunk results
        approximation_depth: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
        sum_fields : bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        facets: list
            Fields to include in the facets, if [] then all
        filters: list
            Query for filtering the search results
        min_score: int
            Minimum score for similarity metric
        include_vector: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        asc: bool
            Whether to sort results by ascending or descending order
        keep_search_history: bool
            Whether to store the history into VecDB. This will increase the storage costs over time.
        hundred_scale: bool
            Whether to scale up the metric by 100
        query: string
            What to store as the query name in the dashboard

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            results = df.chunk_search(
                chunk_field="_chunk_",
                multivector_query=MULTIVECTOR_QUERY
            )

        """
        return self.services.search.chunk(
            dataset_id=self.dataset_id,
            multivector_query=multivector_query,
            chunk_field=chunk_field,
            chunk_scoring=chunk_scoring,
            chunk_page_size=chunk_page_size,
            chunk_page=chunk_page,
            approximation_depth=approximation_depth,
            sum_fields=sum_fields,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            facets=facets,
            filters=filters,
            min_score=min_score,
            include_vector=include_vector,
            include_count=include_count,
            asc=asc,
            keep_search_history=keep_search_history,
            hundred_scale=hundred_scale,
            query=query,
        )

    @track
    def multistep_chunk_search(
        self,
        multivector_query,
        first_step_multivector_query,
        chunk_field,
        chunk_scoring="max",
        chunk_page_size: int = 3,
        chunk_page: int = 1,
        approximation_depth: int = 0,
        sum_fields: bool = True,
        page_size: int = 20,
        page: int = 1,
        similarity_metric: str = "cosine",
        facets: list = [],
        filters: list = [],
        min_score: int = None,
        include_vector: bool = False,
        include_count: bool = True,
        asc: bool = False,
        keep_search_history: bool = False,
        hundred_scale: bool = False,
        first_step_page: int = 1,
        first_step_page_size: int = 20,
        query: str = None,
    ):
        """
        Multistep chunk search involves a vector search followed by chunk search, used to accelerate chunk searches or to identify context before delving into relevant chunks. e.g. Search against the paragraph vector first then sentence chunkvector after. \n

        For more information about chunk search check out services.search.chunk. \n

        For more information about vector search check out services.search.vector

        Parameters
        ----------

        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        chunk_field : string
            Field where the array of chunked documents are.
        chunk_scoring: string
            Scoring method for determining for ranking between document chunks.
        chunk_page_size: int
            Size of each page of chunk results
        chunk_page: int
            Page of the chunk results
        approximation_depth: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
        sum_fields : bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        facets: list
            Fields to include in the facets, if [] then all
        filters: list
            Query for filtering the search results
        min_score: int
            Minimum score for similarity metric
        include_vector: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        asc: bool
            Whether to sort results by ascending or descending order
        keep_search_history: bool
            Whether to store the history into VecDB. This will increase the storage costs over time.
        hundred_scale: bool
            Whether to scale up the metric by 100
        first_step_multivector_query: list
            Query for advance search that allows for multiple vector and field querying.
        first_step_page: int
            Page of the results
        first_step_page_size: int
            Size of each page of results
        query: string
            What to store as the query name in the dashboard

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            results = df.search.multistep_chunk(
                chunk_field="_chunk_",
                multivector_query=MULTIVECTOR_QUERY,
                first_step_multivector_query=FIRST_STEP_MULTIVECTOR_QUERY
            )

        """
        return self.services.search.multistep_chunk(
            dataset_id=self.dataset_id,
            multivector_query=multivector_query,
            first_step_multivector_query=first_step_multivector_query,
            chunk_field=chunk_field,
            chunk_scoring=chunk_scoring,
            chunk_page_size=chunk_page_size,
            chunk_page=chunk_page,
            approximation_depth=approximation_depth,
            sum_fields=sum_fields,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            facets=facets,
            filters=filters,
            min_score=min_score,
            include_vector=include_vector,
            include_count=include_count,
            asc=asc,
            keep_search_history=keep_search_history,
            hundred_scale=hundred_scale,
            first_step_page=first_step_page,
            first_step_page_size=first_step_page_size,
            query=query,
        )

    @track
    def auto_reduce_dimensions(
        self,
        alias: str,
        vector_fields: list,
        filters: Optional[list] = None,
        number_of_documents: Optional[int] = None,
    ):
        """
        Run dimensionality reduction quickly on a dataset on a small number of documents.
        This is useful if you want to quickly see a projection of your dataset.
        Currently, the only supported algorithm is `PCA`.

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems.


        .. note::
            **New in v0.32.0**

        Parameters
        ----------
        vector_fields: list
            The vector fields to run dimensionality reduction on
        number_of_documents: int
            The number of documents to get
        algorithm: str
            The algorithm to run. The only supported algorithm is `pca` at this
            current point in time.
        n_components: int
            The number of components

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            df.auto_reduce_dimensions(
                "pca-3",
                ["sample_vector_"],
            )

        """
        if len(vector_fields) > 1:
            raise ValueError("We only support 1 vector field at the moment.")

        dr_args = alias.split("-")

        if len(dr_args) != 2:
            raise ValueError("""Your DR alias should be in the form of `pca-3`.""")

        algorithm = dr_args[0]
        n_components = int(dr_args[1])

        print("Getting documents...")
        if filters is None:
            filters = []
        filters += [
            {
                "field": vf,
                "filter_type": "exists",
                "condition": ">=",
                "condition_value": " ",
            }
            for vf in vector_fields
        ]

        if number_of_documents is None:
            number_of_documents = self.get_number_of_documents(self.dataset_id, filters)

        documents = self.get_documents(
            select_fields=vector_fields,
            filters=filters,
            number_of_documents=number_of_documents,
        )

        print("Run PCA...")
        if algorithm == "pca":
            dr_documents = self._run_pca(
                vector_fields=vector_fields,
                documents=documents,
                alias=alias,
                n_components=n_components,
            )
        else:
            raise ValueError("DR algorithm not supported.")

        results = self.update_documents(self.dataset_id, dr_documents)

        if n_components == 3:
            projector_url = f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/projector"
            print(f"You can now view your projector at {projector_url}")

        return results

    @track
    def reduce_dimensions(
        self,
        vector_fields: list,
        alias: str,
        number_of_documents: int = 1000,
        algorithm: str = "pca",
        n_components: int = 3,
        filters: list = [],
    ):
        """
        Run dimensionality reduction quickly on a dataset on a small number of documents.
        This is useful if you want to quickly see a projection of your dataset.
        Currently, the only supported algorithm is `PCA`.

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems.


        .. note::
            **New in v0.32.0**

        Parameters
        ----------
        vector_fields: list
            The vector fields to run dimensionality reduction on
        number_of_documents: int
            The number of documents to get
        algorithm: str
            The algorithm to run. The only supported algorithm is `pca` at this
            current point in time.
        n_components: int
            The number of components

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            df.auto_reduce_dimensions(
                alias="pca-3",
                ["sample_vector_"],
                number_of_documents=1000
            )

        """
        if len(vector_fields) > 1:
            raise ValueError("We only support 1 vector field at the moment.")

        print("Getting documents...")
        if filters is None:
            filters = []
        filters += [
            {
                "field": vf,
                "filter_type": "exists",
                "condition": ">=",
                "condition_value": " ",
            }
            for vf in vector_fields
        ]
        documents = self.get_documents(
            select_fields=vector_fields,
            filters=filters,
            number_of_documents=number_of_documents,
        )

        print("Run PCA...")
        if algorithm == "pca":
            dr_documents = self._run_pca(
                vector_fields=vector_fields,
                documents=documents,
                alias=alias,
                n_components=n_components,
            )

        else:
            raise ValueError(
                "DR algorithm not supported. Only supported algorithms are `pca`."
            )

        results = self.update_documents(self.dataset_id, dr_documents)

        return results

    def _run_pca(
        self, vector_fields: list, documents: list, alias: str, n_components: int = 3
    ):
        from relevanceai.vector_tools.dim_reduction import PCA

        model = PCA()
        # Returns a list of documents with the dr vector
        return model.fit_transform_documents(
            vector_field=vector_fields[0],
            documents=documents,
            alias=alias,
            dims=n_components,
        )

    @track
    def auto_cluster(self, alias: str, vector_fields: List[str], chunksize: int = 1024):
        """
        Automatically cluster in 1 line of code.
        It will retrieve documents, run fitting on the documents and then
        update the database.
        There are only 2 supported clustering algorithms at the moment:
        - kmeans
        - minibatchkmeans

        In order to choose the number of clusters, simply add a number
        after the dash like `kmeans-8` or `minibatchkmeans-50`.

        Under the hood, it uses scikit learn defaults or best practices.

        Parameters
        ----------
        alias : str
            The clustering model (as a str) to use and n_clusters. Delivered in a string separated by a '-'
            Supported aliases at the moment are 'kmeans','kmeans-10', 'kmeans-X' (where X is a number), 'minibatchkmeans',
                'minibatchkmeans-10', 'minibatchkmeans-X' (where X is a number)
        vector_fields : List
            A list vector fields over which to cluster

        Example
        ----------

        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            # run kmeans with default 10 clusters
            clusterer = df.auto_cluster("kmeans", vector_fields=[vector_field])
            clusterer.list_closest_to_center()

            # Run k means clustering with 8 clusters
            clusterer = df.auto_cluster("kmeans-8", vector_fields=[vector_field])

            # Run minibatch k means clustering with 8 clusters
            clusterer = df.auto_cluster("minibatchkmeans-8", vector_fields=[vector_field])

            # Run minibatch k means clustering with 20 clusters
            clusterer = df.auto_cluster("minibatchkmeans-20", vector_fields=[vector_field])

        """
        cluster_args = alias.split("-")
        algorithm = cluster_args[0]
        if len(cluster_args) > 1:
            n_clusters = int(cluster_args[1])
        else:
            print("No clusters are detected, defaulting to 8")
            n_clusters = 8
        if n_clusters >= chunksize:
            raise ValueError("Number of clustesr exceed chunksize.")

        num_documents = self.get_number_of_documents(self.dataset_id)

        if num_documents <= n_clusters:
            warnings.warn(
                "You seem to have more clusters than documents. We recommend reducing the number of clusters."
            )

        from relevanceai.clusterer import ClusterOps

        if algorithm.lower() == "kmeans":
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=n_clusters)
            clusterer: ClusterOps = ClusterOps(
                model=model,
                alias=alias,
                api_key=self.api_key,
                project=self.project,
                firebase_uid=self.firebase_uid,
                dataset_id=self.dataset_id,
                vector_fields=vector_fields,
            )
            clusterer.fit_predict_update(dataset=self, vector_fields=vector_fields)

        elif algorithm.lower() == "hdbscan":
            raise ValueError(
                "HDBSCAN is soon to be released as an alternative clustering algorithm"
            )
        elif algorithm.lower() == "minibatchkmeans":
            from sklearn.cluster import MiniBatchKMeans

            model = MiniBatchKMeans(n_clusters=n_clusters)

            clusterer = ClusterOps(
                model=model,
                alias=alias,
                api_key=self.api_key,
                project=self.project,
                firebase_uid=self.firebase_uid,
                dataset_id=self.dataset_id,
                vector_fields=vector_fields,
            )

            clusterer.partial_fit_predict_update(
                dataset=self, vector_fields=vector_fields, chunksize=chunksize
            )
        else:
            raise ValueError("Only KMeans clustering is supported at the moment.")

        return clusterer

    @track
    def aggregate(
        self,
        groupby: list = [],
        metrics: list = [],
        filters: list = [],
        sort: list = [],
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
        alias: str = "default",
    ):
        return self.services.aggregate.aggregate(
            dataset_id=self.dataset_id,
            groupby=groupby,
            metrics=metrics,
            filters=filters,
            page_size=page_size,
            page=page,
            asc=asc,
            flatten=flatten,
            alias=alias,
            # sort=sort
        )

    def facets(
        self,
        fields: list = [],
        date_interval: str = "monthly",
        page_size: int = 5,
        page: int = 1,
        asc: bool = False,
    ):
        return self.datasets.facets(
            dataset_id=self.dataset_id,
            fields=fields,
            date_interval=date_interval,
            page_size=page_size,
            page=page,
            asc=asc,
        )
