"""
Clusterer class to run clustering.
"""
import numpy as np
from relevanceai.api.client import BatchAPIClient
from relevanceai.vector_tools.cluster import Cluster
from typing import Union, List, Dict
from relevanceai.dataset_api import Dataset
from cluster_base import ClusterBase
from doc_utils import DocUtils
from abc import abstractmethod

class ClusterFlow(BatchAPIClient):
    """ClusterFlow class allows users to be able to 
    """
    def __init__(self, 
        model: ClusterBase,
        alias: str,
        cluster_field: str="_cluster_",
        project: str=None,
        api_key: str=None,
    ):
        self.alias = alias
        self.cluster_field = cluster_field
        self.model = model
        self.project = project
        self.api_key = api_key
    
    def _init_dataset(self, dataset):
        if isinstance(dataset, Dataset):
            self.dataset_id = self.dataset.dataset_id
            self.dataset: Dataset = dataset
        else:
            self.dataset_id = dataset
            self.dataset = Dataset(
                project=self.project,
                api_key=self.api_key
            )
    
    def fit(
        self,
        dataset: Union[Dataset, str],
        vector_fields: List,
    ):
        return self.fit_dataset(dataset, vector_fields=vector_fields)
    
    def fit_dataset(
        self,
        dataset: Union[Dataset, str],
        vector_fields: List,
        filters: List=[]
    ):
        """Fit a dataset
        """

        # load the documents
        self.logger.warning(
            "Retrieving documents... This can take a while if the dataset is large."
        )

        self._init_dataset(dataset)
        self.vector_fields = vector_fields

        docs = self.get_all_documents(
            dataset_id=self.dataset_id, 
            filters=filters,
            select_fields=vector_fields
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
        results = self.update_documents(
            self.dataset_id, clustered_docs, chunksize=10000
        )
        self.logger.info(results)

        # Update the centroid collection
        self.model.vector_fields = vector_fields

        self._insert_centroid_documents()
    
    def _insert_centroid_documents(self):
        if hasattr(self.model, "get_centroid_documents"):
            if len(self.vector_fields) == 1:
                centers = self.model.get_centroid_documents(self.vector_fields[0])
            else:
                centers = self.model.get_centroid_documents()

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
    
    # def list_closest_to_center(self):
    #     return self.datasets.cluster.centroids.list_closest_to_center(
    #         dataset_id=self.dataset_id,
    #         vector_fields=self.vector_fields
    #         alias=self.alias
    #     )
    
    def list_furthest_from_center(self):
        """
            Listing Furthest from center
        """
        raise NotImplementedError


    def _concat_vectors_from_list(self, list_of_vectors: list):
        """Concatenate 2 vectors together in a pairwise fashion"""
        return [np.concatenate(x) for x in list_of_vectors]
    
    def _get_vectors_from_documents(self, vector_fields: list,
        documents: List[Dict]):
        if len(vector_fields) == 1:
            # filtering out entries not containing the specified vector
            documents= list(filter(DocUtils.list_doc_fields, documents))
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

        """
        self.vector_fields = vector_fields

        vectors = self._get_vectors_from_documents(vector_fields, documents)

        cluster_labels = self.model.fit_transform(vectors)

        # Label the clusters
        cluster_labels = self._label_clusters(cluster_labels)

        return self.set_cluster_labels_across_documents(cluster_labels, documents, inplace=inplace,
            return_only_clusters=return_only_clusters)
        
    
    def set_cluster_labels_across_documents(self, cluster_labels: list, documents: List[Dict],
        inplace: bool=True, return_only_clusters: bool=True):
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
                {"_id": d.get("_id"), self.cluster_field: d.get(self.cluster_field)} for d in new_documents
            ]
        return new_documents

    def _set_cluster_labels_across_documents(self, cluster_labels, documents):
        if isinstance(self.vector_fields, list):
            set_cluster_field = f"{self.cluster_field}.{'.'.join(self.vector_fields)}.{self.alias}"
        elif isinstance(self.vector_fields, str):
            set_cluster_field = f"{self.cluster_field}.{self.vector_fields}.{self.alias}"
        self.set_field_across_documents(set_cluster_field, cluster_labels, documents)


    def _label_cluster(self, label: Union[int, str]):
        if isinstance(label, (int, float)):
            return "cluster-" + str(label)
        return str(label)

    def _label_clusters(self, labels):
        return [self._label_cluster(x) for x in labels]
