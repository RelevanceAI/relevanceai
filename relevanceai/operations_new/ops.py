"""
RelevanceAI Operations wrappers for use from a Dataset object

.. code-block::

    from relevanceai import Client

    client = Client()

    dataset = client.Dataset()

    dataset.vectorize_text(*args, **kwargs)

    dataset.reduce_dims(*args, **kwargs)

    dataset.cluster(*args **kwarsgs)
"""

from typing import Any, Dict, List, Optional

from relevanceai.dataset.write import Write
from relevanceai.utils.decorators.analytics import track


class Operations(Write):
    @track
    def reduce_dims(
        self,
        vector_fields: List[str],
        n_components: int = 3,
        batched: bool = True,
        model: Optional[Any] = None,
        model_kwargs: Optional[dict] = None,
        alias: Optional[str] = None,
        filters: Optional[list] = None,
        chunksize: Optional[int] = 100,
    ):
        """It takes a list of fields, a list of models, a list of filters, and a chunksize, and then runs
        the DimReductionOps class on the documents in the dataset

        Parameters
        ----------
        fields : List[str]
            List[str]
        models : Optional[List[Any]]
            List[Any] = None,
        filters : Optional[List[Dict[str, Any]]]
            A list of dictionaries, each dictionary containing a filter.
        chunksize : int, optional
            The number of documents to process at a time.

        Returns
        -------
            Nothing is being returned.

        """
        from relevanceai.operations_new.dr.ops import DimReductionOps

        model = "pca" if model is None else model

        ops = DimReductionOps(
            credentials=self.credentials,
            vector_fields=vector_fields,
            n_components=n_components,
            model=model,
            model_kwargs=model_kwargs,
            alias=alias,
        )

        res = ops.run(
            dataset=self,
            select_fields=vector_fields,
            chunksize=chunksize,
            filters=filters,
            batched=batched,
        )

        return ops

    @track
    def vectorize_text(
        self,
        fields: List[str],
        batched: bool = True,
        models: Optional[List[Any]] = None,
        filters: Optional[list] = None,
        chunksize: Optional[int] = 20,
    ):
        """It takes a list of fields, a list of models, a list of filters, and a chunksize, and then it runs
        the VectorizeOps function on the documents in the database

        Parameters
        ----------
        fields : List[str]
            List[str]
        models : List[Any]
            List[Any]
        filters : List[Dict[str, Any]]
            List[Dict[str, Any]]
        chunksize : int, optional
            int = 100,

        Returns
        -------
            Nothing

        """
        from relevanceai.operations_new.vectorize.text.ops import VectorizeTextOps

        models = ["all-mpnet-base-v2"] if models is None else models

        ops = VectorizeTextOps(
            credentials=self.credentials,
            fields=fields,
            models=models,
        )

        res = ops.run(
            dataset=self,
            select_fields=fields,
            filters=filters,
            batched=batched,
            chunksize=chunksize,
        )

        return ops

    @track
    def vectorize_image(
        self,
        fields: List[str],
        models: Optional[List[Any]] = None,
        batched: Optional[bool] = True,
        filters: Optional[list] = None,
        chunksize: Optional[int] = None,
    ):
        """It takes a list of fields, a list of models, a list of filters, and a chunksize, and then it runs
        the VectorizeOps function on the documents in the database

        Parameters
        ----------
        fields : List[str]
            List[str]
        models : List[Any]
            List[Any]
        filters : List[Dict[str, Any]]
            List[Dict[str, Any]]
        chunksize : int, optional
            int = 100,

        Returns
        -------
            Nothing

        """
        from relevanceai.operations_new.vectorize.image.ops import VectorizeImageOps

        models = ["clip"] if models is None else models

        ops = VectorizeImageOps(
            credentials=self.credentials,
            fields=fields,
            models=models,
        )

        res = ops.run(
            dataset=self,
            select_fields=fields,
            filters=filters,
            batched=batched,
            chunksize=chunksize,
        )

        return ops

    @track
    def label(
        self,
        vector_fields: List[str],
        label_documents: List[Any],
        expanded: bool = True,
        max_number_of_labels: int = 1,
        similarity_metric: str = "cosine",
        similarity_threshold: float = 0,
        label_field: str = "label",
        label_vector_field: str = "label_vector_",
        batched: bool = False,
        filters: Optional[list] = None,
        chunksize: Optional[int] = 100,
    ):
        """This function takes a list of documents, a list of vector fields, and a list of label documents,
        and then it labels the documents with the label documents

        Parameters
        ----------
        vector_fields : List[str]
            List[str]
        label_documents : List[Any]
            List[Any]
        expanded, optional
            If True, the label_vector_field will be a list of vectors. If False, the label_vector_field
        will be a single vector.
        max_number_of_labels : int, optional
            int = 1,
        similarity_metric : str, optional
            str = "cosine",
        filters : Optional[list]
            A list of filters to apply to the documents.
        chunksize : int, optional
            The number of documents to process at a time.
        similarity_threshold : float, optional
            float = 0,
        label_field : str, optional
            The name of the field that will contain the label.
        label_vector_field, optional
            The field that will be added to the documents that contain the label vector.

        Returns
        -------
            The return value is a list of documents.

        """

        from relevanceai.operations_new.label.ops import LabelOps

        ops = LabelOps(
            credentials=self.credentials,
            vector_field=vector_fields[0],
            expanded=expanded,
            max_number_of_labels=max_number_of_labels,
            similarity_metric=similarity_metric,
            similarity_threshold=similarity_threshold,
            label_field=label_field,
            label_vector_field=label_vector_field,
        )

        res = ops.run(
            dataset=self,
            label_documents=label_documents,
            filters=filters,
            batched=batched,
            chunksize=chunksize,
        )

        return ops

    @track
    def split_sentences(
        self,
        text_fields: List[str],
        output_field="_splittextchunk_",
        language: str = "en",
        inplace: bool = True,
        batched: bool = False,
        filters: Optional[list] = None,
        chunksize: Optional[int] = 100,
    ):
        """
        This function splits the text in the `text_field` into sentences and stores the sentences in
        the `output_field`

        Parameters
        ----------
        text_field : str
            The field in the documents that contains the text to be split into sentences.
        output_field, optional
            The name of the field that will contain the split sentences.
        language : str, optional
            The language of the text. This is used to determine the sentence splitting rules.

        """
        from relevanceai.operations_new.processing.text.sentence_splitting.ops import (
            SentenceSplitterOps,
        )

        ops = SentenceSplitterOps(
            credentials=self.credentials,
            text_fields=text_fields,
            language=language,
            inplace=inplace,
            output_field=output_field,
        )

        res = ops.run(
            dataset=self,
            select_fields=text_fields,
            batched=batched,
            filters=filters,
            chunksize=chunksize,
        )

        return ops

    @track
    def cluster(
        self,
        vector_fields: List[str],
        model: Optional[Any] = None,
        alias: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        chunksize: Optional[int] = 100,
        filters: Optional[list] = None,
        batched: Optional[bool] = False,
        include_cluster_report: bool = True,
        **kwargs,
    ):
        """`cluster` is a function that takes in a list of vector fields, a model, an alias, a list of
        filters, a boolean value, a dictionary of model keyword arguments, and a list of keyword
        arguments. It returns an object of type `ClusterOps`

        Example
        ----------
        .. code-block::

            from sklearn.cluster import KMeans
            model = KMeans()

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            cluster_ops = ds.cluster(
                model=model, vector_fields=["sample_vector_"],
                alias="kmeans-8"
            )

        Parameters
        ----------
        vector_fields : List[str]
            A list of possible vector fields
        model : Optional[Any]
            The clustering model to use. Currently, we support KMeans and MiniBatchKMeans.
        alias : Optional[str]
            The name of the cluster model.
        filters : Optional[list]
            Optional[list] = None,
        include_cluster_report : bool, optional
            bool = True
        model_kwargs : Optional[Dict[str, Any]]
            The cluster config to use
            You can change the number of clusters for kmeans using:
            `cluster_config={"n_clusters": 10}`. For a full list of
            possible parameters for different models, simply check how
            the cluster models are instantiated.

        Returns
        -------
            The cluster object

        """

        from relevanceai.operations_new.cluster.ops import ClusterOps

        model = "kmeans" if model is None else model
        model_kwargs = {} if model_kwargs is None else model_kwargs

        ops = ClusterOps(
            model=model,
            alias=alias,  # type: ignore
            vector_fields=vector_fields,  # type: ignore
            verbose=False,
            credentials=self.credentials,
            dataset_id=self.dataset_id,
            model_kwargs=model_kwargs,
            **kwargs,
        )

        if filters is not None:
            filters = ops._get_filters(filters, vector_fields)

        # Create the cluster report
        ops.run(
            dataset=self,
            select_fields=vector_fields,
            batched=batched,
            chunksize=chunksize,
            filters=filters,
        )

        print(
            f"""You can now utilise the ClusterOps object using the below:

    cluster_ops = client.ClusterOps(
        alias='{ops.alias}',
        vector_fields={ops.vector_fields},
        dataset_id='{self.dataset_id}'
    )"""
        )

        print("Configure your new cluster app below:")
        print(
            f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster/"
        )
        return ops

    def _get_alias(self, alias: Any) -> str:
        # Auto-generates alias here
        if alias is None:
            if hasattr(self.model, "n_clusters"):
                n_clusters = (
                    self.n_clusters
                    if self.n_clusters is not None
                    else self.model.n_clusters
                )
                alias = f"{self.model_name}-{n_clusters}"

            elif hasattr(self.model, "k"):
                n_clusters = (
                    self.n_clusters if self.n_clusters is not None else self.model.k
                )
                alias = f"{self.model_name}-{n_clusters}"

            else:
                alias = self.model_name

            Warning.MISSING_ALIAS.format(alias=alias)  # type: ignore

        if self.verbose:
            print(f"The alias is `{alias.lower()}`.")
        return alias.lower()

    @track
    def batch_cluster(
        self,
        vector_fields: List[str],
        model: Any = None,
        alias: Optional[str] = None,
        filters: Optional[list] = None,
        include_cluster_report: bool = True,
        model_kwargs: dict = None,
        **kwargs,
    ):
        from relevanceai.operations_new.cluster.batch.ops import BatchClusterOps

        cluster_ops = BatchClusterOps(
            model=model,
            alias=alias,
            vector_fields=vector_fields,
            model_kwargs=model_kwargs,
            **kwargs,
        )

        if filters is not None:
            filters = cluster_ops._get_filters(filters, vector_fields)
        cluster_ops.run(self, filters)
        return cluster_ops
