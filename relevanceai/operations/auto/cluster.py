# -*- coding: utf-8 -*-
"""
Pandas like dataset API
"""
import warnings
import numpy as np

from typing import List, Optional
from tqdm.auto import tqdm

from relevanceai.utils.decorators.analytics import track
from relevanceai.utils.decorators.version import beta
from relevanceai.constants.warning import Warning

# from relevanceai.operations.cluster.models.community_detection import CommunityDetection

# class AutoClusterOps(CommunityDetection):
#     @track
#     def cluster(self, model, alias, vector_fields, **kwargs):
#         """
#         Performs KMeans Clustering on over a vector field within the dataset.

#         .. warning::
#             Deprecated in v0.33 in favour of df.auto_cluster.

#         Parameters
#         ------------
#         model : Class
#             The clustering model to use
#         vector_fields : str
#             The vector fields over which to cluster

#         Example
#         ------------
#         .. code-block::

#             from relevanceai import Client
#             from relevanceai ClusterOps
#             from relevanceai.ops.clusterops.kmeans_clusterer import KMeansModel

#             client = Client()

#             dataset_id = "sample_dataset_id"
#             df = client.Dataset(dataset_id)

#             vector_field = "vector_field_"
#             n_clusters = 10

#             model = KMeansModel(k=n_clusters)

#             df.cluster(model=model, alias=f"kmeans-{n_clusters}", vector_fields=[vector_field])
#         """
#         from relevanceai.operations.cluster import ClusterOps

#         clusterer = ClusterOps(
#             credentials=self.credentials,
#             model=model,
#             alias=alias,
#         )
#         clusterer.fit(dataset=self, vector_fields=vector_fields)
#         return clusterer

#     def cluster_keyphrases(
#         self,
#         vector_fields: List[str],
#         text_fields: List[str],
#         cluster_alias: str,
#         cluster_field: str = "_cluster_",
#         num_clusters: int = 100,
#         most_common: int = 10,
#         preprocess_hooks: Optional[List[callable]] = None,
#         algorithm: str = "rake",
#         n: int = 2,
#     ):
#         """
#         Simple implementation of the cluster word cloud.

#         Parameters
#         ------------
#         vector_fields: list
#             The list of vector fields
#         text_fields: list
#             The list of text fields
#         cluster_alias: str
#             The alias of the cluster
#         cluster_field: str
#             The cluster field to try things on
#         num_clusters: int
#             The number of clusters
#         preprocess_hooks: list
#             The preprocess hooks
#         algorithm: str
#             The algorithm to use
#         n: int
#             The number of words

#         """
#         preprocess_hooks = [] if preprocess_hooks is None else preprocess_hooks

#         vector_fields_str = ".".join(sorted(vector_fields))
#         field = f"{cluster_field}.{vector_fields_str}.{cluster_alias}"
#         all_clusters = self.facets([field], page_size=num_clusters)
#         cluster_counters = {}
#         if "results" in all_clusters:
#             all_clusters = all_clusters["results"]
#         # TODO: Switch to multiprocessing
#         for c in tqdm(all_clusters[field]):
#             cluster_value = c[field]
#             top_words = self.keyphrases(
#                 text_fields=text_fields,
#                 n=n,
#                 filters=[
#                     {
#                         "field": field,
#                         "filter_type": "contains",
#                         "condition": "==",
#                         "condition_value": cluster_value,
#                     }
#                 ],
#                 most_common=most_common,
#                 preprocess_hooks=preprocess_hooks,
#                 algorithm=algorithm,
#             )
#             cluster_counters[cluster_value] = top_words
#         return cluster_counters

#     # TODO: Add keyphrases to auto cluster
#     def auto_cluster_keyphrases(
#         vector_fields: List[str],
#         text_fields: List[str],
#         cluster_alias: str,
#         deployable_id: str,
#         n: int = 2,
#         cluster_field: str = "_cluster_",
#         num_clusters: int = 100,
#         preprocess_hooks: Optional[List[callable]] = None,
#     ):
#         """
#         # TODO:
#         """
#         pass

#     def _add_cluster_word_cloud_to_config(self, data, cluster_value, top_words):
#         # TODO: Add this to wordcloud deployable
#         # hacky way I implemented to add top words to config
#         data["configuration"]["cluster-labels"][cluster_value] = ", ".join(
#             [k for k in top_words if k != "machine learning"]
#         )
#         data["configuration"]["cluster-descriptions"][cluster_value] = str(top_words)

#     @track
#     def auto_cluster(
#         self,
#         alias: str,
#         vector_fields: List[str],
#         model=None,
#         chunksize: int = 1024,
#         filters: Optional[list] = None,
#         parent_alias: Optional[str] = None,
#     ):
#         """
#         Automatically cluster in 1 line of code.
#         It will retrieve documents, run fitting on the documents and then
#         update the database.
#         There are only 2 supported clustering algorithms at the moment:
#         - kmeans
#         - minibatchkmeans

#         In order to choose the number of clusters, simply add a number
#         after the dash like `kmeans-8` or `minibatchkmeans-50`.

#         Under the hood, it uses scikit learn defaults or best practices.

#         This returns a ClusterOps object and is a wrapper on top of
#         `ClusterOps`.

#         Parameters
#         ----------
#         alias : str
#             The clustering model (as a str) to use and n_clusters. Delivered in a string separated by a '-'
#             Supported aliases at the moment are 'kmeans','kmeans-10', 'kmeans-X' (where X is a number), 'minibatchkmeans',
#                 'minibatchkmeans-10', 'minibatchkmeans-X' (where X is a number)
#         vector_fields : List
#             A list vector fields over which to cluster

#         Example
#         ----------

#         .. code-block::

#             from relevanceai import Client

#             client = Client()

#             dataset_id = "sample_dataset_id"
#             df = client.Dataset(dataset_id)

#             # run kmeans with default 10 clusters
#             clusterer = df.auto_cluster("kmeans", vector_fields=[vector_field])
#             clusterer.list_closest()

#             # Run k means clustering with 8 clusters
#             clusterer = df.auto_cluster("kmeans-8", vector_fields=[vector_field])

#             # Run minibatch k means clustering with 8 clusters
#             clusterer = df.auto_cluster("minibatchkmeans-8", vector_fields=[vector_field])

#             # Run minibatch k means clustering with 20 clusters
#             clusterer = df.auto_cluster("minibatchkmeans-20", vector_fields=[vector_field])

#         You can alternatively run this using kmeans.

#         .. code-block::

#             from relevanceai import Client

#             client = Client()

#             from relevanceai.package_utils.datasets import mock_documents

#             ds = client.Dataset('sample')
#             ds.upsert_documents(mock_documents(100))
#             # Run initial kmeans to get clusters
#             ds.auto_cluster('kmeans-3', vector_fields=["sample_1_vector_"])
#             # Run separate K Means to get subclusters
#             cluster_ops = ds.auto_cluster(
#                 'kmeans-2',
#                 vector_fields=["sample_1_vector_"],
#                 parent_alias="kmeans-3"
#             )

#         """
#         if model is None:
#             return self._auto_cluster_string(
#                 alias=alias,
#                 vector_fields=vector_fields,
#                 chunksize=1024,
#                 filters=filters,
#                 parent_alias=parent_alias,
#             )
#         else:
#             return self._auto_cluster_model(
#                 alias=alias,
#                 model=model,
#                 vector_fields=vector_fields,
#                 chunksize=chunksize,
#                 filters=filters,
#                 parent_alias=parent_alias,
#             )

#     def _auto_cluster_model(
#         self,
#         alias: str,
#         vector_fields: List[str],
#         chunksize: int = 1024,
#         filters: Optional[list] = None,
#         parent_alias: Optional[str] = None,
#         model=None,
#     ):
#         filters = [] if filters is None else filters

#         cluster_args = alias.split("-")
#         algorithm = cluster_args[0]
#         if len(cluster_args) > 1:
#             n_clusters = int(cluster_args[1])
#         else:
#             print("No clusters are detected, defaulting to 8")
#             n_clusters = 8
#         if n_clusters >= chunksize:
#             raise ValueError("Number of clustesr exceed chunksize.")

#         num_documents = self.get_number_of_documents(self.dataset_id)

#         if num_documents <= n_clusters:
#             warnings.warn(Warning.NCLUSTERS_GREATER_THAN_NDOCS)

#         from relevanceai.operations.cluster import ClusterOps

#         clusterer: ClusterOps = ClusterOps(
#             credentials=self.credentials,
#             model=model,
#             alias=alias,
#         )
#         if parent_alias:
#             clusterer.subcluster_predict_update(
#                 dataset=self,
#                 vector_fields=vector_fields,
#                 filters=filters,
#             )
#         else:
#             clusterer.fit(
#                 dataset_id=self.dataset_id,
#                 vector_fields=vector_fields,
#                 parent_alias=parent_alias,
#                 include_grade=True,
#                 filters=filters,
#             )
#         return clusterer

#     def _store_subcluster_metadata(
#         self, vector_fields: list, alias: str, parent_alias: str
#     ):
#         # Store metadata around subclustering
#         field = str("-".join(vector_fields)) + "." + alias
#         metadata = self.metadata
#         if "subcluster" not in metadata:
#             metadata["subclusters"] = []
#         metadata["subclusters"].append(
#             {
#                 "parent_alias": parent_alias,
#                 "alias": alias,
#                 "vector_fields": vector_fields,
#             }
#         )
#         self.upsert_metadata(metadata)

#     @track
#     def _auto_cluster_string(
#         self,
#         alias: str,
#         vector_fields: List[str],
#         chunksize: int = 1024,
#         filters: Optional[list] = None,
#         parent_alias: Optional[str] = None,
#     ):
#         """
#         Automatically cluster in 1 line of code.
#         It will retrieve documents, run fitting on the documents and then
#         update the database.
#         There are only 2 supported clustering algorithms at the moment:
#         - kmeans
#         - minibatchkmeans

#         In order to choose the number of clusters, simply add a number
#         after the dash like `kmeans-8` or `minibatchkmeans-50`.

#         Under the hood, it uses scikit learn defaults or best practices.

#         This returns a ClusterOps object and is a wrapper on top of
#         `ClusterOps`.

#         Parameters
#         ----------
#         alias : str
#             The clustering model (as a str) to use and n_clusters. Delivered in a string separated by a '-'
#             Supported aliases at the moment are 'kmeans','kmeans-10', 'kmeans-X' (where X is a number), 'minibatchkmeans',
#                 'minibatchkmeans-10', 'minibatchkmeans-X' (where X is a number)
#         vector_fields : List
#             A list vector fields over which to cluster

#         Example
#         ----------

#         .. code-block::

#             from relevanceai import Client

#             client = Client()

#             dataset_id = "sample_dataset_id"
#             df = client.Dataset(dataset_id)

#             # run kmeans with default 10 clusters
#             clusterer = df.auto_cluster("kmeans", vector_fields=[vector_field])
#             clusterer.list_closest()

#             # Run k means clustering with 8 clusters
#             clusterer = df.auto_cluster("kmeans-8", vector_fields=[vector_field])

#             # Run minibatch k means clustering with 8 clusters
#             clusterer = df.auto_cluster("minibatchkmeans-8", vector_fields=[vector_field])

#             # Run minibatch k means clustering with 20 clusters
#             clusterer = df.auto_cluster("minibatchkmeans-20", vector_fields=[vector_field])

#         You can alternatively run this using kmeans.

#         .. code-block::

#             from relevanceai import Client

#             client = Client()

#             from relevanceai.package_utils.datasets import mock_documents

#             ds = client.Dataset('sample')
#             ds.upsert_documents(mock_documents(100))
#             # Run initial kmeans to get clusters
#             ds.auto_cluster('kmeans-3', vector_fields=["sample_1_vector_"])
#             # Run separate K Means to get subclusters
#             cluster_ops = ds.auto_cluster(
#                 'kmeans-2',
#                 vector_fields=["sample_1_vector_"],
#                 parent_alias="kmeans-3"
#             )

#         """
#         filters = [] if filters is None else filters

#         cluster_args = alias.split("-")
#         algorithm = cluster_args[0]
#         if len(cluster_args) > 1:
#             n_clusters = int(cluster_args[1])
#         else:
#             print("No clusters are detected, defaulting to 8")
#             n_clusters = 8
#         if n_clusters >= chunksize:
#             raise ValueError("Number of clustesr exceed chunksize.")

#         num_documents = self.get_number_of_documents(self.dataset_id)

#         if num_documents <= n_clusters:
#             warnings.warn(
#                 "You seem to have more clusters than documents. We recommend reducing the number of clusters."
#             )

#         from relevanceai.operations.cluster import ClusterOps

#         if algorithm.lower() == "kmeans":
#             from sklearn.cluster import KMeans

#             model = KMeans(n_clusters=n_clusters)
#             clusterer: ClusterOps = ClusterOps(
#                 credentials=self.credentials,
#                 model=model,
#                 alias=alias,
#                 dataset_id=self.dataset_id,
#                 vector_fields=vector_fields,
#                 parent_alias=parent_alias,
#             )
#             if parent_alias:
#                 clusterer.subcluster_predict_update(
#                     dataset=self,
#                     vector_fields=vector_fields,
#                     filters=filters,
#                 )

#                 self._store_subcluster_metadata(
#                     vector_fields=vector_fields, alias=alias, parent_alias=parent_alias
#                 )
#             else:
#                 clusterer.fit(
#                     dataset=self,
#                     vector_fields=vector_fields,
#                     include_report=True,
#                     filters=filters,
#                 )

#         elif algorithm.lower() == "hdbscan":
#             raise ValueError(
#                 "HDBSCAN is soon to be released as an alternative clustering algorithm"
#             )
#         elif algorithm.lower() == "minibatchkmeans":
#             from sklearn.cluster import MiniBatchKMeans

#             model = MiniBatchKMeans(n_clusters=n_clusters)

#             clusterer = ClusterOps(
#                 credentials=self.credentials,
#                 model=model,
#                 alias=alias,
#                 dataset_id=self.dataset_id,
#                 vector_fields=vector_fields,
#                 parent_alias=parent_alias,
#             )

#             if parent_alias:
#                 print("subpartial fit...")
#                 clusterer.subpartialfit_predict_update(
#                     dataset=self,
#                     vector_fields=vector_fields,
#                     filters=filters,
#                 )

#                 self._store_subcluster_metadata(
#                     vector_fields=vector_fields, alias=alias, parent_alias=parent_alias
#                 )

#             else:
#                 clusterer.partial_fit_predict_update(
#                     dataset=self,
#                     vector_fields=vector_fields,
#                     chunksize=chunksize,
#                     filters=filters,
#                 )

#         elif algorithm.lower() == "communitydetection":
#             if len(vector_fields) > 1:
#                 raise ValueError(
#                     "Currently we do not support more than 1 vector field."
#                 )
#             return self.community_detection(field=vector_fields[0], alias=alias)
#         else:
#             raise ValueError("Only KMeans clustering is supported at the moment.")

#         return clusterer

#     def auto_text_cluster_dashboard(
#         self,
#         text_fields: List[str],
#         alias: str,
#         chunksize: int = 1024,
#         filters: Optional[list] = None,
#         text_encoder=None,
#     ):
#         """
#         Convenient way to vectorize and cluster text fields.

#         Parameters
#         ----------
#         text_fields: List[str]
#             A list of text fields to vectorize and cluster

#         alias: str
#             The name of the clustring application. The alias is required to
#             be of the form "{algorithm}-{n_clusters}" where:
#                 * algorithm is the clustering algorithm to be used; and
#                 * n_clusters is the number of clusters

#         chunksize: int
#             The size of the chunks

#         filters: Optional[list]
#             A list of filters to apply over the fields to vectorize

#         text_encoder:
#             A deep learning text encoder from the vectorhub library. If no
#             encoder is specified, a default encoder (USE2Vec) is loaded.

#         Returns
#         -------
#         A dictionary indicating the outcome status

#         Example
#         -------

#         .. code-block::

#             from relevanceai import Client

#             client = Client()

#             ds = client.Dataset("sample_dataset_id")

#             ds.auto_text_cluster_dashboard(text_fields=["sample_text_field"])

#         """
#         filters = [] if filters is None else filters

#         fields = []
#         for field in text_fields:
#             try:
#                 if not "text" == self.schema[field]:
#                     fields.append(field)
#             except KeyError:
#                 raise KeyError(f"'{field}' is an invalid field")
#         else:
#             if fields:
#                 raise ValueError(
#                     "The following fields are not text fields: " f"{', '.join(fields)}"
#                 )

#         results = self.vectorize(text_fields=text_fields, text_encoder=text_encoder)
#         if "added_vectors" not in results:
#             # If there were errors in vectorizing, then quit immediately and return errors
#             return results

#         new_vectors = results["added_vectors"]
#         existing_vectors = results["skipped_vectors"]

#         self.auto_cluster(
#             alias=alias,
#             vector_fields=new_vectors + existing_vectors,
#             chunksize=chunksize,
#             filters=filters,
#         )

#         print(
#             "Build your clustering app here: "
#             f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
#         )

#     def launch_cluster_app(self, configuration: dict = None):
#         """
#         Launch an app with a given configuration


#         Example
#         --------

#         .. code-block::

#             ds.launch_cluster_app()

#         Parameters
#         -----------

#         configuration: dict
#             The configuration can be found in the deployable once created.

#         """
#         if configuration is None:
#             url = f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
#             print(
#                 "Build your clustering app here: "
#                 f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
#             )
#             return url
#         if "configuration" in configuration:
#             configuration = configuration["configuration"]
#         results = self.deployables.create(
#             dataset_id=self.dataset_id, configuration=configuration
#         )

#         # After you have created an app
#         url = f"https://cloud.relevance.ai/dataset/{results['dataset_id']}/deploy/cluster/{self.project}/{self.api_key}/{results['deployable_id']}/{self.region}"
#         print(f"You can now access your deployable at {url}.")
#         return url
