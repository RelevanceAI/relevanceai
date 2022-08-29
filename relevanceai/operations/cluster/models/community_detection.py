from relevanceai.dataset.write.write import Write


class CommunityDetection(Write):
    pass


#     @track
#     def community_detection(
#         self,
#         field: str,
#         model=None,
#         retrieval_kwargs: Optional[dict] = None,
#         encode_kwargs: Optional[dict] = None,
#         threshold: float = 0.75,
#         min_community_size: int = 3,
#         init_max_size: int = 1000,
#         update_chunksize: int = 100,
#         alias: str = "community-detection",
#         log_file: str = "logs.txt",
#         cluster_field: str = "_cluster_",
#     ):
#         """
#         Performs community detection on a text field.

#         Parameters
#         ----------
#         field: str
#             The field over which to find communities. Must be of type "text"
#             or "vector".

#         model
#             A model for computing sentence embeddings.

#         retrieval_kwargs: Optional[dict]
#             Keyword arguments for `get_documents` call. See respective
#             details for argument details.

#         encode_kwargs: Optional[dict]
#             Keyword arguments for the provide model's `encode` call. See
#             respective method for argument details.

#         threshold: float
#             A lower limit of similarity that determines whether two embeddings
#             are similar or not.

#         min_community_size: int
#             The minimum size of a community. Only communities that are larger
#             than this value are returned, and the first element of each
#             community is treated as the central point.

#         init_max_size: int
#             The maximum size of a community. If the corpus is larger than this
#             value, that is set to the maximum size.

#         Example
#         -------

#         .. code-block::

#             from relevanceai import Client

#             client = Client()

#             ds = client.Dataset("sample_dataset_id")

#             communities = ds.community_detection("sample_text_field")

#         """
#         if field in self.schema:
#             if not (self.schema[field] == "text" or field.endswith("_vector_")):
#                 raise ValueError("The field must be a 'text' type or a vector")
#         else:
#             raise ValueError(f"{field} does not exist in the dataset")

#         try:
#             with FileLogger(log_file):
# from sentence_transformers.util import community_detection
#         except ModuleNotFoundError:
#             raise ModuleNotFoundError(
#                 "community_detection function not found. "
#                 "Please install sentence-transformers with `python -m "
#                 "pip install -U sentence-transformers` to install "
#                 "community_detection."
#             )

#         field_type = "text" if self.schema[field] == "text" else "vector"

#         retrieval_kwargs = {} if retrieval_kwargs is None else retrieval_kwargs
#         encode_kwargs = {} if encode_kwargs is None else encode_kwargs

#         if model is None and field_type == "text":
#             from sentence_transformers import SentenceTransformer

#             model = SentenceTransformer("all-MiniLM-L6-v2")
#             # encode defaults:
#             #  chunksize: int = 32
#             #  show_progress_bar: bool = None
#             #  output_value: str = 'sentence_embedding'
#             #  convert_to_numpy: bool = True
#             #  convert_to_Tensor: bool = False
#             #  device: str = None
#             #  normalize_embeddings: bool = False

#         print("Retrieving documents...")
#         documents = self.get_all_documents(
#             select_fields=[field],
#             filters=[
#                 {
#                     "field": field,
#                     "filter_type": "exists",
#                     "condition": "==",
#                     "condition_value": " ",
#                 }
#             ],
#             **{
#                 key: value
#                 for key, value in retrieval_kwargs.items()
#                 if key != "select_fields"
#             },
#         )
#         print("Documents retrieved.")

#         # lists (i.e. vectors) are unhashable therefore handled by converting
#         # them into tuples.
#         if field_type == "vector":
#             for document in documents:
#                 try:
#                     value = tuple(self.get_field(field, document))
#                     self.set_field(field, document, value)
#                 except KeyError:
#                     # If a document is missing a vector, ignore
#                     continue

#         # Keep track of the fields. Since two documents could have the same
#         # field, use a list to keep track of multiples
#         element_ids = defaultdict(list)

#         # remove duplicates
#         elements = set()
#         for document in documents:
#             try:
#                 element = self.get_field(field, document)
#             except Exception as e:
#                 # It could be that a document does not have a field or a
#                 # a vector that other documents in the Dataset has. In that
#                 # case, ignore.
#                 traceback.print_exc()
#                 continue
#             elements.add(element)
#             element_ids[element].append(document["_id"])

#         # Storing a mapping like this is fine because when the values are
#         # made a list below, they will be a list in the same order as inserted
#         # in the dictionary.
#         element_map = {}
#         ids_map = {}
#         for i, element in enumerate(elements):
#             element_map[i] = element
#             ids_map[i] = element_ids[element]

#         if field_type == "text":
#             print("Encoding the corpus...")
#             embeddings = model.encode(
#                 sentences=list(element_map.values()),
#                 **{
#                     key: value
#                     for key, value in encode_kwargs.items()
#                     if key != "sentences"
#                 },
#             )
#             print("Encoding complete.")
#         else:
#             from numpy import array

#             embeddings = array(list(element_map.values()))

#         print("Community detection started...")
#         init_max_size = min(init_max_size, embeddings.shape[0])
#         clusters = community_detection(
#             embeddings=embeddings,
#             threshold=threshold,
#             min_community_size=min_community_size,
#             init_max_size=init_max_size,
#         )
#         print(f"Detected {len(clusters)} communities.")

#         # TODO: add centroids for community detection

#         print("Updating documents...")
#         community_documents = []
#         centroids: List[dict] = []
#         for i, cluster in enumerate(tqdm(clusters)):
#             ids = []
#             centroid = cluster[0]
#             centroids.append(
#                 {
#                     "_id": ids_map[centroid][0],  # choose the first ID as centroid
#                     "centroid_vector_": embeddings[centroid].tolist(),
#                 }
#             )
#             for member in cluster:
#                 ids.extend(ids_map[member])
#             # During initial construction update_where did not accept dict
#             # values as valid updates.
#             # self.datasets.documents.update_where(
#             #    self.dataset_id,
#             #    update={
#             #        "_cluster_": {field: {"community-detection": f"community-{i+1}"}}
#             #    },
#             #    filters=[
#             #        {
#             #            "field": "ids",
#             #            "filter_type": "ids",
#             #            "condition": "==",
#             #            "condition_value": ids,
#             #        }
#             #    ],
#             # )
#             for id in ids:
#                 community_documents.append(
#                     {
#                         "_id": id,
#                         cluster_field: {field: {alias: f"cluster-{i+1}"}},
#                     }
#                 )

#             # During initial construction update_where did not accept dict
#             # values as valid updates.
#             with FileLogger(log_file):
#                 result = self.datasets.documents.update_where(
#                     self.dataset_id,
#                     update={"_cluster_": {field: {alias: f"cluster-{i+1}"}}},
#                     filters=[
#                         {
#                             "field": "ids",
#                             "filter_type": "ids",
#                             "condition": "==",
#                             "condition_value": ids,
#                         }
#                     ],
#                 )
#                 print(result)

#         print(
#             "Build your clustering app here: "
#             f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
#         )
#         # Return a ClusterOps object

#         from relevanceai.workflows.cluster_ops.ops import ClusterOps

#         cluster_ops: ClusterOps = ClusterOps(
#             # model=model,
#             alias=alias,
#             dataset_id=self.dataset_id,
#             vector_fields=[field],
#             cluster_field=cluster_field,
#             project=self.project,
#             api_key=self.api_key,
#             firebase_uid=self.firebase_uid,
#             verbose=False,
#         )
#         print("Creating centroids...")
#         with FileLogger(log_file):
#             result = cluster_ops.insert_centroid_documents(centroids)
#         print("✅ Uploaded centroids.")
#         return cluster_ops
