"""
Pandas like dataset API
"""
from typing import Dict, List
from relevanceai.dataset_api.dataset_write import Write
from relevanceai.dataset_api.dataset_series import Series
from relevanceai.vector_tools.nearest_neighbours import (
    NearestNeighbours,
    NEAREST_NEIGHBOURS,
)


class Operations(Write):
    def vectorize(self, field: str, model):
        """
        Vectorizes a Particular field (text) of the dataset

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems.

        Parameters
        ----------
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

            dataset_id = "sample_dataset"
            df = client.Dataset(dataset_id)

            text_field = "text_field"
            df.vectorize(text_field, model)
        """
        return Series(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            field=field,
        ).vectorize(model)

    def cluster(self, model, alias, vector_fields, **kwargs):
        """
        Performs KMeans Clustering on over a vector field within the dataset.

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
            from relevanceai.clusterer import Clusterer
            from relevanceai.clusterer.kmeans_clusterer import KMeansModel

            client = Client()

            dataset_id = "sample_dataset"
            df = client.Dataset(dataset_id)

            vector_field = "vector_field_"
            n_clusters = 10

            model = KMeansModel(k=n_clusters)

            df.cluster(model=model, alias=f"kmeans-{n_clusters}", vector_fields=[vector_field])
        """
        from relevanceai.clusterer import Clusterer

        clusterer = Clusterer(
            model=model, alias=alias, api_key=self.api_key, project=self.project
        )
        clusterer.fit(dataset=self, vector_fields=vector_fields)
        return clusterer


class LabelExperiment(Operations):
    def label_vector(
        self,
        vector,
        alias: str,
        label_dataset: str,
        label_vector_field: str,
        label_fields: list,
        number_of_labels: int = 1,
        similarity_metric: NEAREST_NEIGHBOURS = "cosine",
        score_field: str = "_search_score",
        **kwargs
    ):
        """
        Label a dataset based on a model.

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems.

        Parameters
        -------------

        vector_fields: list
            The list of vector field
        label_dataset: str
            The dataset to label with
        alias: str
            The alias of the labels (for example - "ranking_labels")
        label_dataset: str
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
        """
        # Download documents in the label dataset
        label_documents: list = self._get_all_documents(
            label_dataset, select_fields=[label_vector_field] + label_fields
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

    def store_labels_in_document(self, labels: list, alias: str):
        # return {"_label_": {label_vector_field: {alias: labels}}}
        return {"_label_": {alias: labels}}

    def _get_nearest_labels(
        self,
        label_documents: List[Dict],
        vector: List[float],
        label_vector_field: str,
        similarity_metric: NEAREST_NEIGHBOURS,
        number_of_labels: int,
        score_field: str,
        label_fields: List[str],
    ):
        nearest_neighbors: List[Dict] = NearestNeighbours.get_nearest_neighbours(
            label_documents,
            vector,
            label_vector_field,
            similarity_metric,
            score_field="_search_score",
        )[:number_of_labels]
        labels = self.subset_documents([score_field] + label_fields, nearest_neighbors)
        return labels

    def label_document(
        self,
        document: dict,
        vector_field: str,
        vector: List[float],
        alias: str,
        label_dataset: str,
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

        Parameters
        -------------

        vector_fields: list
            The list of vector field
        label_dataset: str
            The dataset to label with
        alias: str
            The alias of the labels (for example - "ranking_labels")
        label_dataset: str
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
        """
        vector = self.get_field(vector_field, document)
        labels = self.label_vector(
            vector_field=vector_field,
            vector=vector,
            alias=alias,
            label_dataset=label_dataset,
            label_vector_field=label_vector_field,
            label_fields=label_fields,
            number_of_labels=number_of_labels,
            score_field=score_field,
            similarity_metric=similarity_metric,
        )
        document.update(self.store_labels_in_document(labels, alias))
        return document

    def label(
        self,
        vector_field: str,
        alias: str,
        label_dataset: str,
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

        Parameters
        -------------

        vector_field: str
            The vector field to match with
        label_dataset: str
            The dataset to label with
        alias: str
            The alias of the labels (for example - "ranking_labels")
        label_dataset: str
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
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")
            result = df.label_vector(
                generate_random_vector(100),
                label_vector_field="sample_1_vector_",
                alias="sample",
                label_dataset=test_dataset_df.dataset_id,
                label_fields=["path"],
                number_of_labels=1,
            )

        """
        # Download documents in the label dataset
        label_documents: list = self._get_all_documents(
            label_dataset, select_fields=[label_vector_field] + label_fields
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

        return self.bulk_apply(bulk_label_documents)
