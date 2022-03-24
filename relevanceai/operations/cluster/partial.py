from typing import Union, Optional, List, Dict, Any

from relevanceai.utils.decorators.analytics import track

from relevanceai.operations.cluster.utils import _ClusterOps


class PartialClusterOps(_ClusterOps):
    @track
    def partial_fit_documents(
        self,
        vector_fields: List[Any],
        documents: List[Dict],
    ):
        """
        Train clustering algorithm on documents and then store the labels
        inside the documents.

        Parameters
        -----------
        vector_field: list
            The vector field of the documents
        docs: list
            List of documents to run clustering on
        alias: str
            What the clusters can be called
        cluster_field: str
            What the cluster fields should be called
        return_only_clusters: bool
            If True, return only clusters, otherwise returns the original document
        inplace: bool
            If True, the documents are edited inplace otherwise, a copy is made first
        kwargs: dict
            Any other keyword argument will go directly into the clustering algorithm

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")

            from sklearn.cluster import MiniBatchKMeans
            model = MiniBatchKMeans(n_clusters=2)
            cluster_ops = client.ClusterOps(alias="batchkmeans_2", model=model)

            cluster_ops.parital_fit(df, vector_fields=["documentation_vector_"])
            cluster_ops.predict_update(df, vector_fields=["sample_vector_"])

        """
        self.vector_fields = vector_fields

        vectors = self._get_vectors_from_documents(vector_fields, documents)

        self.model.partial_fit(vectors)

    @track
    def partial_fit_dataset(
        self,
        dataset_id: str,
        vector_fields: List[str],
        chunksize: int = 100,
        filters: Optional[list] = None,
    ):
        """
        Fit The dataset by partial documents.


        Example
        --------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")

            from sklearn.cluster import MiniBatchKMeans
            model = MiniBatchKMeans(n_clusters=2)
            cluster_ops = client.ClusterOps(alias="minibatchkmeans_2", model=model)

            cluster_ops.partial_fit_dataset(df, vector_fields=["documentation_vector_"])

        """
        filters = [] if filters is None else filters

        self.vector_fields = vector_fields
        if len(vector_fields) > 1:
            raise ValueError(
                "We currently do not support multiple vector fields on partial fit"
            )

        filters = [
            {
                "field": f,
                "filter_type": "exists",
                "condition": "==",
                "condition_value": " ",
            }
            for f in vector_fields
        ] + filters

        for c in self._chunk_dataset(
            dataset_id, self.vector_fields, chunksize=chunksize, filters=filters
        ):
            vectors = self._get_vectors_from_documents(vector_fields, c)
            self.model.partial_fit(vectors)

    @track
    def partial_fit_predict_update(
        self,
        dataset_id: str,
        vector_fields: Optional[List[str]] = None,
        chunksize: int = 100,
        filters: Optional[List] = None,
        verbose: bool = True,
    ):
        """
        Fit, predict and update on a dataset.
        Users can also start to run these separately one by one.

        Parameters
        --------------

        dataset: Union[Dataset]
            The dataset class

        vector_fields: List[str]
            The list of vector fields

        chunksize: int
            The size of the chunks

        Example
        -----------

        .. code-block::

            # Real-life example from Research Dashboard
            from relevanceai import Client
            client = Client()
            df = client.Dataset("research2vec")

            from sklearn.cluster import MiniBatchKMeans
            model = MiniBatchKMeans(n_clusters=50)
            cluster_ops = client.ClusterOps(alias="minibatchkmeans_50", model=model)

            cluster_ops.partial_fit_predict_update(
                df,
                vector_fields=['title_trainedresearchqgen_vector_'],
                chunksize=1000
            )

        """
        vector_fields = [] if vector_fields is None else vector_fields
        filters = [] if filters is None else filters

        if verbose:
            print("Fitting dataset...")
        self.partial_fit_dataset(
            dataset_id=dataset_id,
            vector_fields=vector_fields,
            chunksize=chunksize,
            filters=filters,
        )
        if verbose:
            print("Updating your dataset...")
        self.predict_update(dataset_id=dataset_id)
        if hasattr(self.model, "get_centers"):
            if verbose:
                print("Inserting your centroids...")
            self.insert_centroid_documents(
                self.get_centroid_documents(), dataset=dataset_id
            )

        if verbose:
            print(
                "Build your clustering app here: "
                + f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
            )
