Core RelevanceAI Specs
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

Admin
----------------------------

.. automodule:: relevanceai.api.admin
   :members:

Aggregate
--------------------------------

.. automodule:: relevanceai.api.aggregate
   :members:

Centroids
--------------------------------

.. automodule:: relevanceai.api.centroids
   :members:

Client
-----------------------------

.. automodule:: relevanceai.api.client
   :members:

Cluster
------------------------------

.. automodule:: relevanceai.api.cluster
   :members:

Document Operations
--------------------------------

.. automodule:: relevanceai.api.documents
   :members:

Available Encoders
-------------------------------

.. automodule:: relevanceai.api.encoders
   :members:

Monitoring
------------------------------

.. automodule:: relevanceai.api.monitor
   :members:

Recommendations
--------------------------------

.. automodule:: relevanceai.api.recommend
   :members:

Request Configuration
---------------------------------------

.. automodule:: relevanceai.api.requests_config
   :members:

Search
-----------------------------

.. automodule:: relevanceai.api.search
   :members:

Tasks
----------------------------

.. automodule:: relevanceai.api.tasks
   :members:

Module contents
---------------

.. automodule:: relevanceai.api
   :members:
