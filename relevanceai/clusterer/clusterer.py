"""
Clusterer class to run clustering. It is intended to be integrated with
models that inherit from `ClusterBase`.

You can run the Clusterer as such:

.. code-block::

    from relevanceai import Client
    from relevanceai.clusterer import KMeansModel
    client = Client()
    model = KMeansModel(n_clusters=2)
    clusterer = client.Clusterer(model, alias="kmeans_2")
    df = client.Dataset("_github_repo_vectorai")
    clusterer.fit(df, ["documentation_vector_"])

You can view other examples of how to interact with this class here :ref:`integration`.

"""
import os
import json
import getpass

import numpy as np

from scipy.spatial import distance
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    silhouette_score,
    homogeneity_score,
)

from relevanceai.api.client import BatchAPIClient
from typing import Union, List, Dict, Optional, Callable
from relevanceai.clusterer.cluster_base import ClusterBase, CentroidClusterBase

# We use the second import because the first one seems to be causing errors with isinstance
# from relevanceai.dataset_api import Dataset
from relevanceai.integration_checks import is_sklearn_available
from relevanceai.dataset_api.cluster_groupby import ClusterGroupby, ClusterAgg
from relevanceai.dataset_api import Dataset

from doc_utils import DocUtils

SILHOUETTE_INFO = """
Good clusters have clusters which are highly seperated and elements within which are highly cohesive. <br/>
<b>Silohuette Score</b> is a metric from <b>-1 to 1</b> that calculates the average cohesion and seperation of each element, with <b>1</b> being clustered perfectly, <b>0</b> being indifferent and <b>-1</b> being clustered the wrong way"""

RANDOM_INFO = """Good clusters have elements, which, when paired, belong to the same cluster label and same ground truth label. <br/>
<b>Rand Index</b> is a metric from <b>0 to 1</b> that represents the percentage of element pairs that have a matching cluster and ground truth labels with <b>1</b> matching perfect and <b>0</b> matching randomly. <br/> <i>Note: This measure is adjusted for randomness so does not equal the exact numerical percentage.</i>"""

HOMOGENEITY_INFO = """Good clusters only have elements from the same ground truth within the same cluster<br/>
<b>Homogeneity</b> is a metric from <b>0 to 1</b> that represents whether clusters contain only elements in the same ground truth with <b>1</b> being perfect and <b>0</b> being absolutely incorrect."""

COMPLETENESS_INFO = """Good clusters have all elements from the same ground truth within the same cluster <br/>
<b>Completeness</b> is a metric from <b>0 to 1</b> that represents whether clusters contain all elements in the same ground truth with <b>1</b> being perfect and <b>0</b> being absolutely incorrect."""

AVERAGE_SCORE = """Averages other metrics by first normalising values between 0 and 1 <br/>
<b>Average</b> is a metric from <b>0 to 1</b> that averages other metrics by first normalising values between 0 and 1."""

METRIC_DESCRIPTION = {
    "silhouette": SILHOUETTE_INFO,
    "random": RANDOM_INFO,
    "homogeneity": HOMOGENEITY_INFO,
    "completeness": COMPLETENESS_INFO,
    "average": AVERAGE_SCORE,
}


class Clusterer(BatchAPIClient):

    _cred_fn = ".creds.json"

    """
    Clusterer class allows users to set up any clustering model to fit on a Dataset.

    You can read about the other parameters here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Example
    -----------

    .. code-block::

        from relevanceai import Client
        from relevanceai.clusterer import KMeansModel

        model = KMeans(n_clusters=2)
        clusterer = client.Clusterer(model, alias="kmeans_2")

        df = client.Dataset("sample")
        clusterer.fit(df, vector_fields=["sample_vector_"])

    """

    def __init__(
        self,
        model: Union[ClusterBase, CentroidClusterBase],
        alias: str,
        project: Union[str, None] = None,
        api_key: Union[str, None] = None,
        cluster_field: str = "_cluster_",
    ):
        self.alias = alias
        self.cluster_field = cluster_field
        self.model = self._assign_model(model)

        if project is None or api_key is None:
            project, api_key = self._token_to_auth()
        else:
            self.project: str = project
            self.api_key: str = api_key

        super().__init__(project=project, api_key=api_key)

    def __call__(self):
        self.groupby = ClusterGroupby(
            self.project,
            self.api_key,
            self.dataset_id,
            alias=self.alias,
            vector_fields=self.vector_fields,
        )
        self.agg = ClusterAgg(
            self.project,
            self.api_key,
            self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
        )

    def groupby(self):
        return ClusterGroupby(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            alias=self.alias,
            vector_fields=self.vector_fields,
        )

    def agg(self, groupby_call):
        """Aggregate the cluster class."""
        self.groupby = ClusterGroupby(
            self.project,
            self.api_key,
            self.dataset_id,
            alias=self.alias,
            vector_fields=self.vector_fields,
        )
        return ClusterAgg(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            groupby_call=groupby_call,
        )

    # Adding first-class sklearn integration
    def _assign_sklearn_model(self, model):
        # Add support for not just sklearn models but sklearn models
        # with first -class integration for kmeans
        from sklearn.cluster import KMeans, MiniBatchKMeans

        if isinstance(model, (KMeans, MiniBatchKMeans)):

            class CentroidClusterModel(CentroidClusterBase):
                def __init__(self, model):
                    self.model = model

                def fit_transform(self, X):
                    return self.model.fit_transform(X)

                def get_centers(self):
                    return self.model.cluster_centers_

            new_model = CentroidClusterModel(model)
            return new_model

        if hasattr(model, "fit_documents"):
            return model
        elif hasattr(model, "fit_transform"):
            data = {"fit_transform": model.fit_transform, "metadata": model.__dict__}
            ClusterModel = type("ClusterBase", (ClusterBase,), data)
            return ClusterModel()
        elif hasattr(model, "fit_predict"):
            data = {"fit_transform": model.fit_predict, "metadata": model.__dict__}
            ClusterModel = type("ClusterBase", (ClusterBase,), data)
            return ClusterModel()

    def _assign_model(self, model):
        # Check if this is a model that will fit
        # otherwise - forces a Clusterbase
        if is_sklearn_available() and "sklearn" in str(type(model)):
            model = self._assign_sklearn_model(model)
            if model is not None:
                return model

        if isinstance(model, ClusterBase):
            return model
        elif hasattr(model, "fit_documents"):
            return model
        elif hasattr(model, "fit_transform"):
            # Support for SKLEARN interface
            data = {"fit_transform": model.fit_transform, "metadata": model.__dict__}
            ClusterModel = type("ClusterBase", (ClusterBase,), data)
            return ClusterModel()
        elif hasattr(model, "fit_predict"):
            data = {"fit_transform": model.fit_predict, "metadata": model.__dict__}
            ClusterModel = type("ClusterBase", (ClusterBase,), data)
            return ClusterModel()
        raise TypeError("Model should be inherited from ClusterBase.")

    def _token_to_auth(self):
        SIGNUP_URL = "https://cloud.relevance.ai/sdk/api"
        if not os.path.exists(self._cred_fn):
            # We repeat it twice because of different behaviours
            print(f"Authorization token (you can find it here: {SIGNUP_URL} )")
            token = getpass.getpass(f"Auth token:")
            project = token.split(":")[0]
            api_key = token.split(":")[1]
            self._write_credentials(project, api_key)
        else:
            data = self._read_credentials()
            project = data["project"]
            api_key = data["api_key"]
        return project, api_key

    def _read_credentials(self):
        return json.load(open(self._cred_fn))

    def _write_credentials(self, project, api_key):
        json.dump({"project": project, "api_key": api_key}, open(self._cred_fn, "w"))

    def _init_dataset(self, dataset):
        if isinstance(dataset, Dataset):
            self.dataset_id = dataset.dataset_id
            self.dataset: Dataset = dataset
        elif isinstance(dataset, str):
            self.dataset_id = dataset
            self.dataset = Dataset(project=self.project, api_key=self.api_key)
        else:
            raise ValueError(
                "Dataset type needs to be either a string or Dataset instance."
            )

    def fit(
        self, dataset: Union[Dataset, str], vector_fields: List, filters: list = []
    ):
        """
        This function takes in the dataset and the relevant vector fields.
        Under the hood, it runs fit_dataset. Sometimes, you may want to modify the behavior
        to adapt it to your needs.

        Parameters
        -------------

        dataset: Union[Dataset, str]
            The dataset to fit the clusterer on
        vector_fields: List[str],
            The vector fields to fit it on

        Example
        ---------
        .. code-block::

            from relevanceai import Client
            from relevanceai.clusterer import KMeansModel
            client = Client()
            model = KMeansModel(n_clusters=2)
            clusterer = client.Clusterer(model, alias="kmeans_2")
            df = client.Dataset("sample_dataset")
            clusterer.fit(df, ["sample_vector_"])

        """
        self.fit_dataset(dataset, vector_fields=vector_fields, filters=filters)
        self._insert_centroid_documents()

    def list_closest_to_center(
        self,
        cluster_ids: List = [],
        centroid_vector_fields: List = [],
        select_fields: List = [],
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: List = [],
        # facets: List = [],
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
    ):
        """
        List of documents closest from the centre.

        Parameters
        ----------
        cluster_ids: lsit
            Any of the cluster ids
        centroid_vector_fields: list
            Vector fields stored
        select_fields: list
            Fields to include in the search results, empty array/list means all fields
        approx: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate
        sum_fields: bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        filters: list
            Query for filtering the search results
        facets: list
            Fields to include in the facets, if [] then all
        min_score: int
            Minimum score for similarity metric
        include_vectors: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        include_facets: bool
            Include facets in the search results

        Example
        --------------
        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")

            from relevanceai.clusterer import KMeansModel
            kmeans = KMeans(n_clusters=5)
            clusterer = client.Clusterer(kmeans)
            clusterer.fit(df, ["sample_vector_"])
            clusterer.list_closest_to_center()

        """
        return self.datasets.cluster.centroids.list_closest_to_center(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            cluster_ids=cluster_ids,
            centroid_vector_fields=centroid_vector_fields,
            select_fields=select_fields,
            approx=approx,
            sum_fields=sum_fields,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            filters=filters,
            min_score=min_score,
            include_vector=include_vector,
            include_count=include_count,
        )

    def aggregate(
        self,
        metrics: list = [],
        sort: list = [],
        groupby: list = [],
        filters: list = [],
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
    ):
        """
        Takes an aggregation query and gets the aggregate of each cluster in a collection. This helps you interpret each cluster and what is in them.
        It can only can be used after a vector field has been clustered. \n

        Aggregation/Groupby of a collection using an aggregation query. The aggregation query is a json body that follows the schema of:

        .. code-block::

            {
                "groupby" : [
                    {"name": <alias>, "field": <field in the collection>, "agg": "category"},
                    {"name": <alias>, "field": <another groupby field in the collection>, "agg": "numeric"}
                ],
                "metrics" : [
                    {"name": <alias>, "field": <numeric field in the collection>, "agg": "avg"}
                    {"name": <alias>, "field": <another numeric field in the collection>, "agg": "max"}
                ]
            }

        For example, one can use the following aggregations to group score based on region and player name.

        .. code-block::

            {
                "groupby" : [
                    {"name": "region", "field": "player_region", "agg": "category"},
                    {"name": "player_name", "field": "name", "agg": "category"}
                ],
                "metrics" : [
                    {"name": "average_score", "field": "final_score", "agg": "avg"},
                    {"name": "max_score", "field": "final_score", "agg": "max"},
                    {'name':'total_score','field':"final_score", 'agg':'sum'},
                    {'name':'average_deaths','field':"final_deaths", 'agg':'avg'},
                    {'name':'highest_deaths','field':"final_deaths", 'agg':'max'},
                ]
            }

        "groupby" is the fields you want to split the data into. These are the available groupby types:

            - category : groupby a field that is a category
            - numeric: groupby a field that is a numeric

        "metrics" is the fields and metrics you want to calculate in each of those, every aggregation includes a frequency metric. These are the available metric types:

            - "avg", "max", "min", "sum", "cardinality"

        The response returned has the following in descending order. \n

        If you want to return documents, specify a "group_size" parameter and a "select_fields" parameter if you want to limit the specific fields chosen. This looks as such:

            .. code-block::

                {
                    'groupby':[
                        {'name':'Manufacturer','field':'manufacturer','agg':'category',
                        'group_size': 10, 'select_fields': ["name"]},
                    ],
                    'metrics':[
                        {'name':'Price Average','field':'price','agg':'avg'},
                    ],
                }

                # ouptut example:
                {"title": {"title": "books", "frequency": 200, "documents": [{...}, {...}]}, {"title": "books", "frequency": 100, "documents": [{...}, {...}]}}

        For array-aggregations, you can add "agg": "array" into the aggregation query.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        metrics: list
            Fields and metrics you want to calculate
        groupby: list
            Fields you want to split the data into
        filters: list
            Query for filtering the search results
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        asc: bool
            Whether to sort results by ascending or descending order
        flatten: bool
            Whether to flatten
        alias: string
            Alias used to name a vector field. Belongs in field_{alias} vector

        Parameters
        ----------
        metrics: list
            Fields and metrics you want to calculate
        groupby: list
            Fields you want to split the data into
        filters: list
            Query for filtering the search results
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        asc: bool
            Whether to sort results by ascending or descending order
        flatten: bool
            Whether to flatten

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")

            from relevanceai.clusterer import KMeansModel
            clusterer = client.Clusterer(5)
            clusterer.fit(df, ["sample_vector_"])
            clusterer.aggregate(
                groupby=[],
                metrics=[
                    {"name": "average_score", "field": "final_score", "agg": "avg"},
                ]
            )


        """
        return self.services.cluster.aggregate(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            groupby=groupby,
            metrics=metrics,
            sort=sort,
            filters=filters,
            alias=self.alias,
            page_size=page_size,
            page=page,
            asc=asc,
            flatten=flatten,
        )

    def list_furthest_from_center(self):
        """
        List of documents furthest from the centre.

        Parameters
        ----------
        cluster_ids: list
            Any of the cluster ids
        select_fields: list
            Fields to include in the search results, empty array/list means all fields
        approx: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate
        sum_fields: bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        filters: list
            Query for filtering the search results
        facets: list
            Fields to include in the facets, if [] then all
        min_score: int
            Minimum score for similarity metric
        include_vectors: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        include_facets: bool
            Include facets in the search results

        Example
        ---------
        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("_github_repo_vectorai")
            from relevanceai.clusterer import KMeansModel
            model = KMeansModel()
            cluster = client.Clusterer(3)
            clusterer.fit(df)
            clusterer.list_furthest_from_center()

        """
        return self.datasets.cluster.centroids.list_furthest_from_center(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
        )

    def _insert_centroid_documents(self):
        if hasattr(self.model, "get_centroid_documents"):
            centers = self.get_centroid_documents()

            if hasattr(self.model, "get_centers"):
                center_vectors = self.model.get_centers()
                centers = self.get_centroid_documents()

            # Change centroids insertion
            results = self.services.cluster.centroids.insert(
                dataset_id=self.dataset_id,
                cluster_centers=centers,
                vector_fields=self.vector_fields,
                alias=self.alias,
            )
            self.logger.info(results)

            self.datasets.cluster.centroids.list_closest_to_center(
                self.dataset_id,
                vector_fields=self.vector_fields,
                alias=self.alias,
                centroid_vector_fields=self.vector_fields,
                page_size=20,
            )
        return

    def get_centroid_documents(self) -> List:
        """
        Get the centroid documents to store. This enables you to use `list_closest_to_center()`
        and `list_furthest_from_center`.

        .. code-block::

            {
                "_id": "document-id-1",
                "centroid_vector_": [0.23, 0.24, 0.23]
            }

        If multiple vector fields returns this:
        Returns multiple

        .. code-block::

            {
                "_id": "document-id-1",
                "blue_vector_": [0.12, 0.312, 0.42],
                "red_vector_": [0.23, 0.41, 0.3]
            }

        """
        self.centers = self.model.get_centers()

        if not hasattr(self, "vector_fields") or len(self.vector_fields) == 1:
            if isinstance(self.centers, np.ndarray):
                self.centers = self.centers.tolist()
            centroid_vector_field_name = self.vector_fields[0]
            return [
                {
                    "_id": self._label_cluster(i),
                    centroid_vector_field_name: self.centers[i],
                }
                for i in range(len(self.centers))
            ]
        # For one or more vectors, separate out the vector fields
        # centroid documents are created using multiple vector fields
        centroid_docs = []
        for i, c in enumerate(self.centers):
            centroid_doc = {"_id": self._label_cluster(i)}
            for j, vf in enumerate(self.vector_fields):
                centroid_doc[vf] = self.centers[i][vf]
            centroid_docs.append(centroid_doc.copy())
        return centroid_docs

    @property
    def centroids(self):
        """
        See your centroids if there are any.
        """
        return self.services.cluster.centroids.list(
            self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            page_size=10000,
            # cursor: str = None,
            include_vector=True,
        )

    def delete_centroids(self):
        """Delete the centroids after clustering."""
        # TODO: Fix delete centroids once its moved over to Node JS
        import requests

        base_url = self.config["api.base_url"]
        response = requests.post(
            base_url + "/services/cluster/centroids/delete",
            headers={"Authorization": self.project + ":" + self.api_key},
            params={
                "dataset_id": "_github_repo_vectorai",
                "vector_field": ["documentation_vector_"],
                "alias": self.alias,
            },
        )
        return response.json()["status"]

    def fit_dataset(
        self, dataset: Union[Dataset, str], vector_fields: List, filters: List = []
    ):
        """
        This function fits a cluster model onto a dataset. It sits under `fit`
        and is usually the only function that runs unless you have a get_centroid_documents
        function.

        Parameters
        ---------------
        dataset: Union[Dataset, str],
            The dataset object to fit it on
        vector_fields: list
            The vector fields to fit the model on
        filters: list
            The filters to run it on
        include_filters_for_vector_fields: bool
            If True, only cluster on those with the vector fields inside it.

        Example
        ---------

        .. code-block::

            from relevanceai import ClusterBase, Client
            import random
            client = Client()

            class CustomClusterModel(ClusterBase):
                def fit_transform(self, X):
                    cluster_labels = [random.randint(0, 100) for _ in range(len(X))]
                    return cluster_labels

            model = CustomClusterModel()
            clusterer = client.Clusterer(model, alias="random")
            df = client.Dataset("_github_repo_vectorai")

            clusterer.fit_dataset(df, vector_fields=["documentation_vector_"])

        """

        # load the documents
        self.logger.warning(
            "Retrieving documents... This can take a while if the dataset is large."
        )

        self._init_dataset(dataset)
        self.vector_fields = vector_fields

        # make sure to only get fields where vector fields exist
        filters += [
            {
                "field": f,
                "filter_type": "exists",
                "condition": "==",
                "condition_value": " ",
            }
            for f in vector_fields
        ]
        docs = self._get_all_documents(
            dataset_id=self.dataset_id, filters=filters, select_fields=vector_fields
        )

        clustered_docs = self.model.fit_documents(
            vector_fields,
            docs,
            alias=self.alias,
            cluster_field=self.cluster_field,
            return_only_clusters=True,
            inplace=False,
        )

        # Updating the db
        results = self._update_documents(
            self.dataset_id, clustered_docs, chunksize=10000
        )
        self.logger.info(results)

        # Update the centroid collection
        self.model.vector_fields = vector_fields

    # def list_closest_to_center(self):
    #     return self.datasets.cluster.centroids.list_closest_to_center(
    #         dataset_id=self.dataset_id,
    #         vector_fields=self.vector_fields
    #         alias=self.alias
    #     )

    def _concat_vectors_from_list(self, list_of_vectors: list):
        """Concatenate 2 vectors together in a pairwise fashion"""
        return [np.concatenate(x) for x in list_of_vectors]

    def _get_vectors_from_documents(self, vector_fields: list, documents: List[Dict]):
        if len(vector_fields) == 1:
            # filtering out entries not containing the specified vector
            documents = list(filter(DocUtils.list_doc_fields, documents))
            vectors = self.get_field_across_documents(
                vector_fields[0], documents, missing_treatment="skip"
            )
        else:
            # In multifield clusering, we get all the vectors in each document
            # (skip if they are missing any of the vectors)
            # Then run clustering on the result
            documents = list(self.filter_docs_for_fields(vector_fields, documents))
            all_vectors = self.get_fields_across_documents(
                vector_fields, documents, missing_treatment="skip_if_any_missing"
            )
            # Store the vector field lengths to de-concatenate them later
            self._vector_field_length: dict = {}
            prev_vf = 0
            for i, vf in enumerate(self.vector_fields):
                self._vector_field_length[vf] = {}
                self._vector_field_length[vf]["start"] = prev_vf
                end_vf = prev_vf + len(all_vectors[0][i])
                self._vector_field_length[vf]["end"] = end_vf
                # Update the ending
                prev_vf = end_vf

            # Store the vector lengths
            vectors = self._concat_vectors_from_list(all_vectors)

        return vectors

    def fit_documents(
        self,
        vector_fields: list,
        documents: List[Dict],
        return_only_clusters: bool = True,
        inplace: bool = True,
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

            from relevanceai import ClusterBase, Client
            import random
            client = Client()

            class CustomClusterModel(ClusterBase):
                def fit_transform(self, X):
                    cluster_labels = [random.randint(0, 100) for _ in range(len(X))]
                    return cluster_labels

            model = CustomClusterModel()
            clusterer = client.Clusterer(model, alias="random")
            df = client.Dataset("_github_repo_vectorai")

            clusterer.fit(df, vector_fields=["documentation_vector_"])

        """
        self.vector_fields = vector_fields

        vectors = self._get_vectors_from_documents(vector_fields, documents)

        cluster_labels = self.model.fit_transform(vectors)

        # Label the clusters
        cluster_labels = self._label_clusters(cluster_labels)

        return self.set_cluster_labels_across_documents(
            cluster_labels,
            documents,
            inplace=inplace,
            return_only_clusters=return_only_clusters,
        )

    def set_cluster_labels_across_documents(
        self,
        cluster_labels: list,
        documents: List[Dict],
        inplace: bool = True,
        return_only_clusters: bool = True,
    ):
        """
        Utility function to allow users to set cluster labels

        Parameters
        ------------
        cluster_labels: List[str, int]
            A list of integers of string. If it is an integer - it will automatically add a 'cluster-' prefix
            to help avoid incorrect data type parsing. You can override this behavior by setting clusters
            as strings.
        documents: List[dict]
            When the documents are in
        inplace: bool
            If True, then the clusters are set in place.
        return_only_clusters: bool
            If True, then the return_only_clusters will return documents with just the cluster field and ID.
            This can be helpful when you want to upsert quickly without having to re-insert the entire document.

        Example
        -----------

        .. code-block::

            labels = list(range(10))
            documents = [{"_id": str(x)} for x in range(10)]
            clusterer.set_cluster_labels_across_documents(labels, documents)

        """
        if inplace:
            self.set_cluster_labels_across_documents(cluster_labels, documents)
            if return_only_clusters:
                return [
                    {"_id": d.get("_id"), self.cluster_field: d.get(self.cluster_field)}
                    for d in documents
                ]
            return documents

        new_documents = documents.copy()

        self.set_cluster_labels_across_documents(cluster_labels, new_documents)
        if return_only_clusters:
            return [
                {"_id": d.get("_id"), self.cluster_field: d.get(self.cluster_field)}
                for d in new_documents
            ]
        return new_documents

    def _set_cluster_labels_across_documents(self, cluster_labels, documents):
        if isinstance(self.vector_fields, list):
            set_cluster_field = (
                f"{self.cluster_field}.{'.'.join(self.vector_fields)}.{self.alias}"
            )
        elif isinstance(self.vector_fields, str):
            set_cluster_field = (
                f"{self.cluster_field}.{self.vector_fields}.{self.alias}"
            )
        self.set_field_across_documents(set_cluster_field, cluster_labels, documents)

    def _label_cluster(self, label: Union[int, str]):
        if isinstance(label, (int, float)):
            return "cluster-" + str(label)
        return str(label)

    def _label_clusters(self, labels):
        return [self._label_cluster(x) for x in labels]

    @property
    def metadata(self):
        """
        If metadata is none, retrieves metadata about a dataset. notably description, data source, etc
        Otherwise, you can store the metadata about your cluster here.

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("_github_repo_vectorai")
            from relevanceai.clusterer import KMeansModel

            model = KMeansModel()
            kmeans = client.Clusterer(model, alias="kmeans_sample")
            kmeans.fit(df, vector_fields=["sample_1_vector_"])
            kmeans.metadata
            # {"k": 10}

        """
        return self.services.cluster.centroids.metadata(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
        )

    @metadata.setter
    def metadata(self, metadata: dict):
        """
        If metadata is none, retrieves metadata about a dataset. notably description, data source, etc
        Otherwise, you can store the metadata about your cluster here.

        Parameters
        ----------
        metadata: dict
           If None, it will retrieve the metadata, otherwise
           it will overwrite the metadata of the cluster

        Example
        ----------

        .. code-block::


            from relevanceai import Client
            client = Client()
            df = client.Dataset("_github_repo_vectorai")
            from relevanceai.clusterer import KMeansModel

            model = KMeansModel()
            kmeans = client.Clusterer(model, alias="kmeans_sample")
            kmeans.fit(df, vector_fields=["sample_1_vector_"])
            kmeans.metadata
            # {"k": 10}

        """
        return self.services.cluster.centroids.metadata(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            metadata=metadata,
        )

    def evaluate(
        self,
        ground_truth_index: Union[str, None] = None,
        metric: Union[str, Callable] = "euclidean",
        random_state: Union[int, None] = None,
        average_score: bool = False,
    ):
        """
        Evaluate your clusters using the silhouette score, or if true labels are provided, using completeness, randomness and homogeniety as well

        Parameters
        ----------
        ground_truth_index : str
            the index of the true label of each sample in dataset

        metric : str or Callable
            A string referencing suportted distance functions, or custom method for calculating the distance between 2 vectors

        random_state : int, default None
            for reproducability

        average_score : bool
            a boolean that determines whether to average the evaluation metrics in a new a score. only applicable if ground_truth_index is not None

        Example
        ----------

        .. code-block::


            from relevanceai import Client
            client = Client()
            df = client.Dataset("_github_repo_vectorai")
            from relevanceai.clusterer import KMeansModel

            model = KMeansModel()
            kmeans = client.Clusterer(model, alias="kmeans_sample")
            kmeans.fit(df, vector_fields=["sample_1_vector_"])

            kmeans.evaluate()
            kmeans.evaluate("truth_index")
        """

        vector_field = self.vector_fields[0]
        cluster_field = self.cluster_field
        alias = self.alias

        samples = self._get_all_documents(self.dataset_id, include_vector=True)

        vectors = [sample[vector_field] for sample in samples]
        pred_labels = [sample[cluster_field][vector_field][alias] for sample in samples]

        def norm(value, min, max):
            return (value - min) / (max - min)

        s_score = silhouette_score(
            vectors, pred_labels, metric=metric, random_state=random_state
        )

        stats = {}
        stats["silhouette"] = {
            "score": s_score,
            "description": METRIC_DESCRIPTION["silhouette"],
        }

        if ground_truth_index:
            ground_truth_labels = {
                key: value[0][ground_truth_index]
                for key, value in self.list_closest_to_center(page_size=1).items()
                if value
            }
            pred_labels = [
                ground_truth_labels[sample[cluster_field][vector_field][alias]]
                for sample in samples
            ]
            true_labels = [sample[ground_truth_index] for sample in samples]

            ar_score = adjusted_rand_score(true_labels, pred_labels)
            c_score = completeness_score(true_labels, pred_labels)
            h_score = homogeneity_score(true_labels, pred_labels)

            stats["random"] = {
                "score": ar_score,
                "description": METRIC_DESCRIPTION["random"],
            }
            stats["completeness"] = {
                "score": c_score,
                "description": METRIC_DESCRIPTION["completeness"],
            }
            stats["homogeneity"] = {
                "score": h_score,
                "description": METRIC_DESCRIPTION["homogeneity"],
            }

            if average_score:
                ns_score = norm(s_score, min=-1, max=1)
                nar_score = norm(ar_score, min=-1, max=1)

                average_score = (ns_score + nar_score + c_score + h_score) / 4

                stats["average"] = {
                    "score": average_score,
                    "description": METRIC_DESCRIPTION["average"],
                }

        return stats
