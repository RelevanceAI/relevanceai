:py:mod:`relevanceai.api.batch.mongo_to_relevance_ai`
=====================================================

.. py:module:: relevanceai.api.batch.mongo_to_relevance_ai

.. autoapi-nested-parse::

   Migrate from mongo database to Relevance Ai:
       #Create an object of Mongo2RelevanceAi class
       connection_string= "..."
       project= "..."
       api_key= "..."
       mongo2vec = Mongo2Mongo2RelevanceAi(connection_string, project, api_key)

       #Get a summary of the mondo database using "mongo_summary"
       mongo2vec.mongo_summary()

       #Set the desired source mongo collection using "set_mongo_collection"
       db_name = '...'
       collection_name = '...'
       mongo2vec.set_mongo_collection(db_name, collection_name)

       #Get total number of entries in the mongo collection using "mongo_doc_count"
       doc_cnt = mongo2vec.mongo_doc_count()

       #Migrate data from mongo to Relevance Ai using "migrate_mongo2relevance_ai"
       chunk_size = 5000      # migrate batches of 5000 (default 2000)
       start_idx= 12000       # loads from mongo starting at index 12000 (default 0)
       dataset_id = "..."     # dataset id in the Relevance Ai platform
       mongo2vec.migrate_mongo2relevance_ai(dataset_id, doc_cnt, chunk_size = chunk_size, start_idx= start_idx)



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.api.batch.mongo_to_relevance_ai.Mongo2RelevanceAi




Attributes
~~~~~~~~~~

.. autoapisummary::

   relevanceai.api.batch.mongo_to_relevance_ai.PYMONGO_AVAILABLE
   relevanceai.api.batch.mongo_to_relevance_ai.BSON_AVAILABLE


.. py:data:: PYMONGO_AVAILABLE
   :annotation: = True

   

.. py:data:: BSON_AVAILABLE
   :annotation: = True

   

.. py:class:: Mongo2RelevanceAi(connection_string: str, project: str, api_key: str)

   Bases: :py:obj:`relevanceai.api.client.BatchAPIClient`

   Batch API client

   .. py:method:: mongo_summary(self)

      returns a dictionary {key:value}
      key = db names
      value = collection names in each db


   .. py:method:: get_mongo_db(self, db_name: str)


   .. py:method:: get_mongo_collection(self, db_name: str, collection_name: str)


   .. py:method:: set_mongo_db(self, db_name: str)


   .. py:method:: set_mongo_collection(self, db_name: str, collection_name: str)


   .. py:method:: mongo_doc_count(self)


   .. py:method:: create_relevance_ai_dataset(self, dataset_id: str)


   .. py:method:: update_id(self, docs: List[dict])


   .. py:method:: parse_json(data)
      :staticmethod:


   .. py:method:: flatten_inner_indxs(docs: List[dict])
      :staticmethod:


   .. py:method:: remove_nan(docs: List[dict], replace_with: str = '')
      :staticmethod:


   .. py:method:: build_range(doc_cnt: int, chunk_size: int = 2000, start_idx: int = 0)
      :staticmethod:


   .. py:method:: fetch_mongo_collection_data(self, start_idx: int = None, end_idx: int = None)


   .. py:method:: migrate_mongo2relevance_ai(self, dataset_id: str, doc_cnt: int, chunk_size: int = 2000, start_idx: int = 0, overwite: bool = False)



