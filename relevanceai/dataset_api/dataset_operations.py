"""
Pandas like dataset API
"""
from typing import Dict, List, Union, Callable, Optional
from relevanceai.dataset_api.dataset_write import Write
from relevanceai.dataset_api.dataset_series import Series
from relevanceai.vector_tools.nearest_neighbours import NearestNeighbours

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
    
    def label(self, vector_fields: list, label_dataset: str, alias: str,
        number_of_labels: int=1, **kwargs):
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
        """
        # Download documents in the label dataset
        label_documents = self.get_all_documents(label_dataset)

        # Build a index
        nearest_neighbors = NearestNeighbours.get_nearest_neighbours(
            label_documents, click_vec, vector_field, distance_measure_mode
        )[:number_of_labels]

        # Store things according to 
        # {"_label_": {"field": {"alias": [{"label": 3, "similarity_score": 0.4}]}

        # Update the original documents
        def bulk_label(docs):
            return docs
