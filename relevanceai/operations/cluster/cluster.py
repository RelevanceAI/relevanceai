import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Callable

from tqdm.auto import tqdm
from relevanceai.client.helpers import Credentials
from relevanceai.constants import CLUSTER_APP_LINK, Warning
from relevanceai._api import APIClient
from relevanceai.utils.decorators.analytics import track
from relevanceai.dataset.dataset import Dataset


class ClusterOps(APIClient):
    def __init__(
        self,
        credentials: Credentials,
        model: Union[str, Any] = None,
        vector_fields: Optional[List[str]] = None,
        alias: Optional[str] = None,
        dataset_id: Optional[str] = None,
        n_clusters: Optional[int] = None,
        cluster_config: Optional[Dict[str, Any]] = None,
        outlier_value: int = -1,
        outlier_label: str = "outlier",
        **kwargs,
    ):
        """
        ClusterOps object

        Parameters
        -------------

        model: Union[str, Any]
            The string of clustering algorithm, class of clustering algorithm or custom clustering class.
            If custom, the model must contain the method for fit_predict and must output a numpy array for labels

        vector_fields: Optional[List[str]] = None
            A list of vector_fields to cluster over

        alias: Optional[str] = None
            An alias to be given for clustering.
            If no alias is provided, alias becomes the cluster model name hypen n_clusters e.g. (kmeans-5, birch-8)

        n_clusters: Optional[int] = None
            Number of clusters to find, should the argument be required of the algorithm.
            If None, n_clusters defaults to 8 for sklearn models ONLY.

        cluster_config: Optional[Dict[str, Any]] = None
            Hyperparameters for the clustering model.
            See the documentation for supported clustering models for list of hyperparameters

        outlier_value: int = -1
            For unsupervised clustering models like OPTICS etc. outliers are given the integer label -1

        outlier_label: str = "outlier"
            When viewing the cluster app dashboard, outliers will be prefixed with outlier_label

        """
        self.vector_field = None if vector_fields is None else vector_fields[0]
        self.vector_fields = vector_fields

        self.config = {} if cluster_config is None else cluster_config  # type: ignore
        if n_clusters is not None:
            self.config["n_clusters"] = n_clusters  # type: ignore

        self.model_name = None
        self.model = self._get_model(model)

        if "package" not in self.__dict__:
            self.package = self._get_package(self.model)

        if hasattr(self.model, "n_clusters"):
            self.n_clusters = self.model.n_clusters
        else:
            self.n_clusters = n_clusters

        self.alias = self._get_alias(alias)
        self.outlier_value = outlier_value
        self.outlier_label = outlier_label
        self.dataset_id = dataset_id

        super().__init__(credentials, **kwargs)

    def __call__(self, dataset_id: str, vector_fields: List[str]) -> None:
        return self.operate(dataset_id=dataset_id, vector_fields=vector_fields)

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

            Warning.MISSING_ALIAS.format(alias=alias)
        return alias.lower()

    def _get_package(self, model):
        model_name = str(model.__class__)
        if "function" in model_name:
            model_name = str(model.__name__)

        if "sklearn" in model_name:
            package = "sklearn"

        elif "faiss" in model_name:
            package = "faiss"

        elif "hdbscan" in model_name:
            package = "hdbscan"

        elif "community_detection" in model_name:
            package = "sentence-transformers"

        else:
            package = "custom"

        return package

    def _get_model(self, model):
        if isinstance(model, str):
            model = model.lower().replace(" ", "").replace("_", "")
            self.model_name = model
        else:
            self.model_name = str(model.__class__).split(".")[-1].split("'>")[0]
            self.package = "custom"
            return model

        if isinstance(model, str):
            if model == "affinitypropagation":
                from sklearn.cluster import AffinityPropagation

                model = AffinityPropagation(**self.config)

            elif model == "agglomerativeclustering":
                from sklearn.cluster import AgglomerativeClustering

                model = AgglomerativeClustering(**self.config)

            elif model == "birch":
                from sklearn.cluster import Birch

                model = Birch(**self.config)

            elif model == "dbscan":
                from sklearn.cluster import DBSCAN

                model = DBSCAN(**self.config)

            elif model == "optics":
                from sklearn.cluster import OPTICS

                model = OPTICS(**self.config)

            elif model == "kmeans":
                from sklearn.cluster import KMeans

                model = KMeans(**self.config)

            elif model == "featureagglomeration":
                from sklearn.cluster import FeatureAgglomeration

                model = FeatureAgglomeration(**self.config)

            elif model == "meanshift":
                from sklearn.cluster import MeanShift

                model = MeanShift(**self.config)

            elif model == "minibatchkmeans":
                from sklearn.cluster import MiniBatchKMeans

                model = MiniBatchKMeans(**self.config)

            elif model == "spectralclustering":
                from sklearn.cluster import SpectralClustering

                model = SpectralClustering(**self.config)

            elif model == "spectralbiclustering":
                from sklearn.cluster import SpectralBiclustering

                model = SpectralBiclustering(**self.config)

            elif model == "spectralcoclustering":
                from sklearn.cluster import SpectralCoclustering

                model = SpectralCoclustering(**self.config)

            elif model == "hdbscan":
                from hdbscan import HDBSCAN

                model = HDBSCAN(**self.config)

            elif model == "community_detection":
                # TODO: this is a callable (?)
                from sentence_transformers.util import community_detection

                model = community_detection

            elif "faiss" in model:
                from faiss import Kmeans

                model = Kmeans(**self.config)

        else:
            # TODO: this needs to be referenced from relevance.constants.errors
            raise ValueError("ModelNotSupported")

        return model

    def _format_labels(self, labels: np.ndarray) -> List[str]:
        labels = labels.flatten().tolist()
        cluster_labels = [
            f"cluster-{str(label)}"
            if label != self.outlier_value
            else self.outlier_label
            for label in labels
        ]
        return cluster_labels

    def _get_centroid_documents(
        self, vectors: np.ndarray, labels: List[str]
    ) -> List[Dict[str, Any]]:
        centroid_documents = []

        centroids: Dict[str, Any] = {}
        for label, vector in zip(labels, vectors.tolist()):
            if label not in centroids:
                centroids[label] = []
            centroids[label].append(vector)

        for centroid, vectors in centroids.items():
            centroid_vector = np.array(vectors).mean(0).tolist()
            centroid_document = dict(
                _id=centroid,
                centroid_vector=centroid_vector,
            )
            centroid_documents.append(centroid_document)

        return centroid_documents

    def _insert_centroids(
        self,
        dataset_id: str,
        vector_field: str,
        centroid_documents: List[Dict[str, Any]],
    ) -> None:
        self.services.cluster.centroids.insert(
            dataset_id=dataset_id,
            cluster_centers=centroid_documents,
            vector_fields=[vector_field],
            alias=self.alias,
        )

    def _fit_predict(
        self, documents: List[Dict[str, Any]], vector_field: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        vectors = np.array(
            [
                self.get_field(vector_field, document)
                for document in documents
                if self.is_field(vector_field, document)
            ]
        )

        if self.package == "sklearn":
            labels = self.model.fit_predict(vectors)

        elif self.package == "hdbscan":
            self.model.fit(vectors)
            labels = self.model.labels_.reshape(-1, 1)

        elif self.package == "faiss":
            vectors = vectors.astype("float32")
            self.model.train(vectors)
            labels = self.model.assign(vectors)[1]

        elif self.package == "sentencetransformers":
            # TODO: make teh config here better
            labels = self.model(vectors)
            labels = np.array(labels)

        elif self.package == "custom":
            labels = self.model.fit_predict(vectors)
            if not isinstance(labels, np.ndarray):
                raise ValueError("Custom model must output np.ndarray for labels")

            if labels.shape[0] != vectors.shape[0]:
                raise ValueError("incorrect shape")

        labels = self._format_labels(labels)

        cluster_field = f"_cluster_.{vector_field}.{self.alias}"
        self.set_field_across_documents(
            field=cluster_field, values=labels, docs=documents
        )

        centroid_documents = self._get_centroid_documents(vectors, labels)

        return centroid_documents, documents

    def _print_app_link(self):
        link = CLUSTER_APP_LINK.format(self.dataset_id)
        print(link)

    def operate(
        self,
        dataset_id: Optional[Union[str, Any]] = None,
        vector_fields: Optional[List[str]] = None,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Run clustering on a dataset
        """
        if not isinstance(dataset_id, str):
            if hasattr(dataset_id, "dataset_id"):
                dataset_id = dataset_id.dataset_id  # type: ignore

        if vector_fields is None:
            vector_fields = self.vector_fields

        self.dataset_id = dataset_id
        if vector_fields is not None:
            vector_field = vector_fields[0]
            self.vector_field = vector_field

        # get all documents
        documents = self._get_all_documents(
            dataset_id=dataset_id,
            select_fields=vector_fields,
            show_progress_bar=show_progress_bar,
            include_vector=True,
        )

        # fit model, predict and label all documents
        centroid_documents, labelled_documents = self._fit_predict(
            documents=documents,
            vector_field=vector_field,
        )

        # TODO: need to change this to an update_where
        self._update_documents(
            dataset_id=dataset_id,
            documents=labelled_documents,
            show_progress_bar=show_progress_bar,
        )

        self._insert_centroids(
            dataset_id=dataset_id,
            vector_field=vector_field,
            centroid_documents=centroid_documents,
        )

        # link back to dashboard
        self._print_app_link()

    def closest(
        self,
        dataset_id: Optional[str] = None,
        vector_field: Optional[str] = None,
        alias: Optional[str] = None,
        **kwargs,
    ):
        """
        List of documents closest from the centre.

        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        vector_field: list
            The vector field where a clustering task was run.
        cluster_ids: list
            Any of the cluster ids
        alias: string
            Alias is used to name a cluster
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

        """
        dataset_id = self.dataset_id if dataset_id is None else dataset_id
        vector_field = self.vector_field if vector_field is None else vector_field
        alias = self.alias if alias is None else alias

        return self.services.cluster.centroids.list_closest_to_center(
            dataset_id=dataset_id, vector_fields=[vector_field], alias=alias, **kwargs
        )

    def furthest(
        self,
        dataset_id: Optional[str] = None,
        vector_field: Optional[str] = None,
        alias: Optional[str] = None,
        **kwargs,
    ):
        """
        List documents furthest from the centre.

        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        vector_fields: list
            The vector field where a clustering task was run.
        cluster_ids: list
            Any of the cluster ids
        alias: string
            Alias is used to name a cluster
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
        """
        dataset_id_ = self.dataset_id if dataset_id is None else dataset_id
        vector_field_ = self.vector_field if vector_field is None else vector_field
        alias_ = self.alias if alias is None else alias

        return self.services.cluster.centroids.list_furthest_from_center(
            dataset_id=dataset_id_,
            vector_fields=[vector_field_],  # type: ignore
            alias=alias_,
            **kwargs,
        )

    # Convenience functions
    list_closest = closest
    list_furthest = furthest

    def _retrieve_dataset_id(self, dataset: Optional[Union[str]]) -> str:
        """Helper method to get multiple dataset values"""
        if isinstance(dataset, Dataset):
            dataset_id: str = dataset.dataset_id
        elif isinstance(dataset, str):
            dataset_id = dataset
        elif dataset is None:
            if hasattr(self, "dataset_id"):
                # let's not surprise users
                print(
                    f"No dataset supplied - using last stored one '{self.dataset_id}'."
                )
                dataset_id = str(self.dataset_id)
            else:
                raise ValueError("Please supply dataset.")
        return dataset_id

    @track
    def aggregate(
        self,
        vector_fields: List[str] = None,
        metrics: Optional[list] = None,
        sort: Optional[list] = None,
        groupby: Optional[list] = None,
        filters: Optional[list] = None,
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
        dataset: Optional[Union[str]] = None,
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
            df = client.Dataset("sample_dataset_id")
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=2)
            cluster_ops = client.ClusterOps(alias="kmeans_2", model=model)
            cluster_ops.fit_predict_update(df, vector_fields=["sample_vector_"])
            clusterer.aggregate(
                "sample_dataset_id",
                groupby=[{
                    "field": "title",
                    "agg": "wordcloud",
                }],
                vector_fields=['sample_vector_']
            )
        """
        metrics = [] if metrics is None else metrics
        sort = [] if sort is None else sort
        groupby = [] if groupby is None else groupby
        filters = [] if filters is None else filters

        return self.services.cluster.aggregate(
            dataset_id=self._retrieve_dataset_id(dataset),
            vector_fields=self.vector_fields if not vector_fields else vector_fields,
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

    @track
    def plot_distributions(
        self,
        numeric_field: str,
        top_indices: int = 10,
        dataset_id: str = None,
    ):
        """
        Plot the sentence length distributions across each cluster
        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("You need to install seaborn! `pip install seaborn`.")
        cluster_field = self._get_cluster_field_name()
        docs = self._get_all_documents(
            dataset_id=dataset_id if dataset_id is None else dataset_id,
            select_fields=[numeric_field, cluster_field],
        )
        df = pd.json_normalize(docs)
        top_comms = df[cluster_field].value_counts()
        for community in top_comms.index[:top_indices]:
            sample_comm_df = df[df[cluster_field] == community]
            sns.displot(sample_comm_df[numeric_field])
            # Get the average in the score too
            mean = sample_comm_df[numeric_field].mean()
            std = sample_comm_df[numeric_field].var()
            plt.title(
                community + str(f" - average: {round(mean, 2)}, var: {round(std, 2)}")
            )
            plt.show()

    @track
    def plot_distributions_by_measure(
        self,
        numeric_field: str,
        measure_function: Callable,
        top_indices: int = 10,
        dataset_id: str = None,
        asc: bool = True,
        measurement_name: str = "measurement",
    ):
        """
        Plot the distributions across each cluster
        measure_function is run on each cluster and plots

        Example
        --------
        .. code-block::
            from scipy.stats import skew
            ops.plot_distributions_measure(numeric_field, skew, dataset_id=dataset_id)
        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("You need to install seaborn! `pip install seaborn`.")
        cluster_field = self._get_cluster_field_name()

        # use the max and min to make the x axis the same
        if dataset_id is None:
            dataset_id = self.dataset_id
        numeric_field_facet = self.datasets.facets(
            dataset_id=dataset_id, fields=[numeric_field]
        )

        facet_result = numeric_field_facet["results"][numeric_field]

        docs = self._get_all_documents(
            dataset_id=dataset_id if dataset_id is None else dataset_id,
            select_fields=[numeric_field, cluster_field],
        )
        df = pd.json_normalize(docs)
        top_comms = df[cluster_field].value_counts()
        cluster_measurements = {}
        for community in tqdm(top_comms.index):
            sample_comm_df = df[df[cluster_field] == community]
            measure_output = measure_function(
                sample_comm_df[numeric_field].dropna().to_list()
            )
            cluster_measurements[community] = measure_output

        cluster_measurements = {
            k: v
            for k, v in sorted(
                cluster_measurements.items(), key=lambda item: item[1], reverse=asc
            )
        }

        for i, (community, measurement) in enumerate(cluster_measurements.items()):
            if i == top_indices:
                return
            sample_comm_df = df[df[cluster_field] == community]
            g = sns.displot(
                sample_comm_df[numeric_field],
            )
            g.set(xlim=(facet_result["min"], facet_result["max"]))
            plt.title(community + str(f" - {measurement_name}: {measurement}"))

    def plot_most_skewed(
        self,
        numeric_field: str,
        top_indices: int = 10,
        dataset_id: str = None,
        asc: bool = True,
    ):
        """
        Plot the skewness of your clusters.
        """
        from scipy.stats import skew

        return self.plot_distributions_by_measure(
            numeric_field=numeric_field,
            measure_function=skew,
            top_indices=top_indices,
            dataset_id=dataset_id,
            asc=asc,
        )

    def _check_for_dataset_id(self):
        if not hasattr(self, "dataset_id"):
            raise ValueError(
                "You are missing a dataset ID. Please set using the argument dataset_id='...'."
            )

    def _get_cluster_field_name(self, alias: str = None):
        if alias is None:
            alias = self.alias
        if isinstance(self.vector_fields, list):
            set_cluster_field = f"_cluster_.{'.'.join(self.vector_fields)}.{alias}"
        elif isinstance(self.vector_fields, str):
            set_cluster_field = f"{self.cluster_field}.{self.vector_fields}.{alias}"
        return set_cluster_field

    def list_cluster_ids(
        self,
        alias: str = None,
        minimum_cluster_size: int = 3,
        dataset_id: str = None,
        num_clusters: int = 1000,
    ):
        """
        List unique cluster IDS

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            cluster_ops = client.ClusterOps(
                alias="kmeans_8", vector_fields=["sample_vector_]
            )
            cluster_ops.list_cluster_ids()
        """
        # Mainly to be used for subclustering
        # Get the cluster alias
        if dataset_id is None:
            self._check_for_dataset_id()
            dataset_id = self.dataset_id

        cluster_field = self._get_cluster_field_name(alias=alias)

        # currently the logic for facets is that when it runs out of pages
        # it just loops - therefore we need to store it in a simple hash
        # and then add them to a list
        all_cluster_ids: Set = set()

        while len(all_cluster_ids) < num_clusters:
            facet_results = self.datasets.facets(
                dataset_id=dataset_id,
                fields=[cluster_field],
                page_size=int(self.config["data.max_clusters"]),
                page=1,
                asc=True,
            )
            if "results" in facet_results:
                facet_results = facet_results["results"]
            if cluster_field not in facet_results:
                raise ValueError(
                    f"No clusters with alias `{alias}`. Please check the schema."
                )
            for facet in facet_results[cluster_field]:
                if facet["frequency"] > minimum_cluster_size:
                    curr_len = len(all_cluster_ids)
                    all_cluster_ids.add(facet[cluster_field])
                    new_len = len(all_cluster_ids)
                    if new_len == curr_len:
                        return list(all_cluster_ids)
        return list(all_cluster_ids)
