Core Features
=======================

The core RelevanceAI specs can be found in this section.

For the RelevanceAI client, we want to ensure the SDK mirrors the API client.

For example:

.. code-block:: python

   ## To instantiate the client 
   from relevanceai import Client
   client = Client()

To use the following endpoint: 

`/datasets/bulk_insert`

You can run: 

.. code-block:: python

   # Bulk insert documents
   client.datasets.bulk_insert(dataset_id, documents)

Or similarly, when you are trying to run 

`/services/search/vector`

You then write: 

.. code-block:: python

   # Vector search in a dataset
   client.services.search.vector(...)

.. automodule:: relevanceai.api.search
   :members:

.. automodule:: relevanceai.api.aggregate
   :members:

.. automodule:: relevanceai.api.recommend
   :members:

.. automodule:: relevanceai.api.admin
   :members:

.. automodule:: relevanceai.api.centroids
   :members:

.. automodule:: relevanceai.api.client
   :members:

.. automodule:: relevanceai.api.cluster
   :members:

.. automodule:: relevanceai.api.documents
   :members:

.. automodule:: relevanceai.api.encoders
   :members:

.. automodule:: relevanceai.api.monitor
   :members:

.. automodule:: relevanceai.api.requests_config
   :members:

.. automodule:: relevanceai.api.tasks
   :members:
