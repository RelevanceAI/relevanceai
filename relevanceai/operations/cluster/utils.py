import os
import json
import warnings
import getpass

import numpy as np

from typing import Union, Callable, Optional, List, Dict, Any, Set

from relevanceai._api import APIClient

from relevanceai.operations.cluster.base import (
    ClusterBase,
    CentroidClusterBase,
    BatchClusterBase,
    HDBSCANClusterBase,
    SklearnCentroidBase,
)

from relevanceai.utils import DocUtils
from relevanceai.utils.integration_checks import (
    is_sklearn_available,
    is_hdbscan_available,
)

from relevanceai.constants.errors import MissingClusterError


class ClusterUtils(APIClient, DocUtils):

    alias: str
    dataset_id: str
    vector_fields: List[Any]

    def _assign_sklearn_model(self, model):
        from sklearn.cluster import (
            KMeans,
            MiniBatchKMeans,
            DBSCAN,
            Birch,
            SpectralClustering,
            OPTICS,
            AgglomerativeClustering,
            AffinityPropagation,
            MeanShift,
            FeatureAgglomeration,
        )

        POSSIBLE_MODELS = [
            SpectralClustering,
            Birch,
            DBSCAN,
            OPTICS,
            AgglomerativeClustering,
            AffinityPropagation,
            MeanShift,
            FeatureAgglomeration,
        ]

        if is_hdbscan_available():
            import hdbscan

            if hasattr(hdbscan, "HDBSCAN"):
                POSSIBLE_MODELS.append(hdbscan.HDBSCAN)

        if model.__class__ == KMeans:

            class CentroidClusterModel(CentroidClusterBase):
                def __init__(self, model):
                    self.model: Union[KMeans, MiniBatchKMeans] = model

                def fit_predict(self, X):
                    return self.model.fit_predict(X)

                def get_centers(self):
                    return self.model.cluster_centers_

            new_model = CentroidClusterModel(model)
            return new_model

        elif model.__class__ == MiniBatchKMeans:

            class BatchCentroidClusterModel(CentroidClusterBase, BatchClusterBase):
                def __init__(self, model):
                    self.model: MiniBatchKMeans = model

                def partial_fit(self, X):
                    return self.model.partial_fit(X)

                def predict(self, X):
                    return self.model.predict(X)

                def get_centers(self):
                    return self.model.cluster_centers_

            new_model = BatchCentroidClusterModel(model)
            return new_model

        elif isinstance(model, tuple(POSSIBLE_MODELS)):
            if "sklearn" in str(type(model)).lower():
                new_model = SklearnCentroidBase(model)

            elif "hdbscan" in str(type(model)).lower():
                new_model = HDBSCANClusterBase(model)

            return new_model

        elif hasattr(model, "fit_documents"):
            return model

        elif hasattr(model, "fit_predict"):
            data = {"fit_predict": model.fit_predict, "metadata": model.__dict__}
            ClusterModel = type("ClusterBase", (ClusterBase,), data)
            return ClusterModel()

        elif hasattr(model, "fit_transform"):
            data = {"fit_predict": model.fit_transform, "metadata": model.__dict__}
            ClusterModel = type("ClusterBase", (ClusterBase,), data)
            return ClusterModel()

    def _assign_model(self, model):

        if (is_sklearn_available() or is_hdbscan_available()) and (
            "sklearn" in str(type(model)).lower()
            or "hdbscan" in str(type(model)).lower()
        ):
            model = self._assign_sklearn_model(model)
            if model is not None:
                return model

        if isinstance(model, ClusterBase):
            return model

        elif hasattr(model, "fit_documents"):
            return model

        elif hasattr(model, "fit_predict"):
            data = {"fit_predict": model.fit_predict, "metadata": model.__dict__}
            ClusterModel = type("ClusterBase", (ClusterBase,), data)
            return ClusterModel()

        elif model is None:
            return model

        raise TypeError("Model should be inherited from ClusterBase.")

    def _token_to_auth(self, token=None):
        SIGNUP_URL = "https://cloud.relevance.ai/sdk/api"

        if os.path.exists(self._cred_fn):
            credentials = self._read_credentials()
            return credentials

        elif token:
            return self._process_token(token)

        else:
            print(f"Activation token (you can find it here: {SIGNUP_URL} )")
            if not token:
                token = getpass.getpass(f"Activation token:")
            return self._process_token(token)

    def _process_token(self, token: str):
        split_token = token.split(":")
        project = split_token[0]
        api_key = split_token[1]
        if len(split_token) > 2:
            region = split_token[3]
            base_url = self._region_to_url(region)

            if len(split_token) > 3:
                firebase_uid = split_token[4]
                return self._write_credentials(
                    project=project,
                    api_key=api_key,
                    base_url=base_url,
                    firebase_uid=firebase_uid,
                )

            else:
                return self._write_credentials(
                    project=project, api_key=api_key, base_url=base_url
                )

        else:
            return self._write_credentials(project=project, api_key=api_key)

    def _read_credentials(self):
        return json.load(open(self._cred_fn))

    def _write_credentials(self, **kwargs):
        print(
            f"Saving credentials to {self._cred_fn}. Remember to delete this file if you do not want credentials saved."
        )
        json.dump(
            kwargs,
            open(self._cred_fn, "w"),
        )
        return kwargs

    def _insert_centroid_documents(self):
        if hasattr(self.model, "get_centroid_documents"):
            print("Inserting centroid documents...")
            centers = self.get_centroid_documents()

            # Change centroids insertion
            results = self.datasets.cluster.centroids.insert(
                dataset_id=self.dataset_id,
                cluster_centers=centers,
                vector_fields=self.vector_fields,
                alias=self.alias,
            )
            self.logger.info(results)

        return

    def _check_for_dataset_id(self):
        if not hasattr(self, "dataset_id"):
            raise ValueError(
                "You are missing a dataset ID. Please set using the argument dataset_id='...'."
            )

    def _concat_vectors_from_list(self, list_of_vectors: list):
        """Concatenate 2 vectors together in a pairwise fashion"""
        return [np.concatenate(x) for x in list_of_vectors]

    def _get_vectors_from_documents(
        self, vector_fields: List[Any], documents: List[Dict]
    ):
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

    def _chunk_dataset(
        self,
        dataset_id: str,
        select_fields: Optional[list] = None,
        chunksize: int = 100,
        filters: Optional[list] = None,
    ):
        """Utility function for chunking a dataset"""
        select_fields = [] if select_fields is None else select_fields
        filters = [] if filters is None else filters

        docs = self._get_documents(
            dataset_id=self.dataset_id,
            include_cursor=False,
            number_of_documents=chunksize,
            select_fields=select_fields,
            filters=filters,
            include_after_id=True,
        )

        while len(docs["documents"]) > 0:
            yield docs["documents"]
            docs = self._get_documents(
                dataset_id=self.dataset_id,
                include_cursor=False,
                select_fields=select_fields,
                number_of_documents=chunksize,
                filters=filters,
                after_id=docs["after_id"],
                include_after_id=True,
            )

    def _get_parent_cluster_values(
        self, vector_fields: list, alias: str, documents
    ) -> list:
        if hasattr(self, "cluster_field"):
            field = ".".join(
                [self.cluster_field, ".".join(sorted(vector_fields)), alias]
            )
        else:
            field = ".".join(["_cluster_", ".".join(sorted(vector_fields)), alias])
        return self.get_field_across_documents(
            field, documents, missing_treatment="skip"
        )

    @staticmethod
    def _calculate_silhouette_grade(vectors, cluster_labels):
        from relevanceai.reports.cluster.grading import get_silhouette_grade
        from sklearn.metrics import silhouette_samples

        score = silhouette_samples(vectors, cluster_labels, metric="euclidean").mean()
        grade = get_silhouette_grade(score)

        print("---------------------------")
        print(f"Grade: {grade}")
        print(f"Mean Silhouette Score: {score}")
        print("---------------------------")

    def _get_cluster_field_name(self, alias: str = None):
        if alias is None:
            alias = self.alias
        if isinstance(self.vector_fields, list):
            if hasattr(self, "cluster_field"):
                set_cluster_field = (
                    f"{self.cluster_field}.{'.'.join(self.vector_fields)}.{alias}"
                )
            else:
                set_cluster_field = f"_cluster_.{'.'.join(self.vector_fields)}.{alias}"
        elif isinstance(self.vector_fields, str):
            set_cluster_field = f"{self.cluster_field}.{self.vector_fields}.{alias}"
        return set_cluster_field

    def _set_cluster_labels_across_documents(self, cluster_labels, documents):
        set_cluster_field = self._get_cluster_field_name()
        self.set_field_across_documents(set_cluster_field, cluster_labels, documents)

    def _label_cluster(self, label: Union[int, str]):
        if not isinstance(label, str):
            return "cluster-" + str(label)
        return str(label)

    def _label_subcluster(self, label: Union[int, str], prev_cluster_label) -> str:
        return prev_cluster_label + "-" + str(label)

    def _label_clusters(self, labels):
        return [self._label_cluster(x) for x in labels]

    def _label_subclusters(self, labels: List[str], prev_cluster_labels: List[str]):
        return [
            self._label_subcluster(label, prev_cluster_label)
            for label, prev_cluster_label in zip(labels, prev_cluster_labels)
        ]

    def _operate_across_clusters(self, field: str, func: Callable):
        output: Dict[str, Any] = dict()
        for cluster_id in self.list_cluster_ids():
            self._operate(cluster_id=cluster_id, field=field, output=output, func=func)
        return output

    def _operate(self, cluster_id: str, field: str, output: dict, func: Callable):
        """
        Internal function for operations
        """
        cluster_field = self._get_cluster_field_name()
        # TODO; change this to fetch all documents
        documents = self.datasets.documents.get_where(
            self.dataset_id,
            filters=[
                {
                    "field": cluster_field,
                    "filter_type": "exact_match",
                    "condition": "==",
                    "condition_value": cluster_id,
                },
                {
                    "field": field,
                    "filter_type": "exists",
                    "condition": ">=",
                    "condition_value": " ",
                },
            ],
            select_fields=[field, cluster_field],
            page_size=9999,
        )
        # get the field across each
        arr = self.get_field_across_documents(field, documents["documents"])
        output[cluster_id] = func(arr)

    def _get_filter_for_cluster(self, cluster_id):
        cluster_field = self._get_cluster_field_name()
        filters = [
            {
                "field": cluster_field,
                "filter_type": "exact_match",
                "condition": "==",
                "condition_value": cluster_id,
            }
        ]
        return filters

    # class ClusterUtilsShow(Read):
    #     def __init__(self, df, is_image_field: bool):
    #         self.df = df
    #         if is_image_field:
    #             self.text_fields = []
    #             self.image_fields = df.columns.tolist()
    #         else:
    #             self.text_fields = df.columns.tolist()
    #             self.image_fields = []

    #     def _repr_html_(self):
    #         try:
    #             documents = self.df.to_dict(orient="records")
    #             return self._show_json(documents, return_html=True)
    #         except Exception as e:
    #             warnings.warn(
    #                 "Displaying using pandas. To get image functionality please install RelevanceAI[notebook]. "
    #                 + str(e)
    #             )
    #             return self.df._repr_html_()

    #     def _show_json(self, documents, **kw):
    #         from jsonshower import show_json

    #         return show_json(
    #             documents,
    #             text_fields=self.text_fields,
    #             image_fields=self.image_fields,
    #             **kw,
    #         )

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

        Parameters
        -------------
        alias: str
            The alias to use for clustering
        minimum_cluster_size: int
            The minimum size of the clusters
        dataset_id: str
            The dataset ID
        num_clusters: int
            The number of clusters

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
                raise MissingClusterError(alias=alias)
            for facet in facet_results[cluster_field]:
                if facet["frequency"] > minimum_cluster_size:
                    curr_len = len(all_cluster_ids)
                    all_cluster_ids.add(facet[cluster_field])
                    new_len = len(all_cluster_ids)
                    if new_len == curr_len:
                        return list(all_cluster_ids)

        return list(all_cluster_ids)

    def list_unique(
        self,
        field: str = None,
        minimum_amount: int = 3,
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
            cluster_ops.list_unique()

        Parameters
        -------------
        alias: str
            The alias to use for clustering
        minimum_cluster_size: int
            The minimum size of the clusters
        dataset_id: str
            The dataset ID
        num_clusters: int
            The number of clusters

        """
        # TODO - should this return a generator?
        # Mainly to be used for subclustering
        # Get the cluster alias
        if dataset_id is None:
            self._check_for_dataset_id()
            dataset_id = self.dataset_id

        # currently the logic for facets is that when it runs out of pages
        # it just loops - therefore we need to store it in a simple hash
        # and then add them to a list
        all_cluster_ids: Set = set()

        if field is None:
            field = self._get_cluster_field_name()

        while len(all_cluster_ids) < num_clusters:
            facet_results = self.datasets.facets(
                dataset_id=dataset_id,
                fields=[field],
                page_size=100,
                page=1,
                asc=True,
            )
            if "results" in facet_results:
                facet_results = facet_results["results"]
            if field not in facet_results:
                raise ValueError("Not valid field. Check schema.")
            for facet in facet_results[field]:
                if facet["frequency"] > minimum_amount:
                    curr_len = len(all_cluster_ids)
                    all_cluster_ids.add(facet[field])
                    new_len = len(all_cluster_ids)
                    if new_len == curr_len:
                        return list(all_cluster_ids)

        return list(all_cluster_ids)
