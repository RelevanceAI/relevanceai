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
   :undoc-members:
   :show-inheritance:

Aggregate
--------------------------------

.. automodule:: relevanceai.api.aggregate
   :members:
   :undoc-members:
   :show-inheritance:

Centroids
--------------------------------

.. automodule:: relevanceai.api.centroids
   :members:
   :undoc-members:
   :show-inheritance:

Client
-----------------------------

.. automodule:: relevanceai.api.client
   :members:
   :undoc-members:
   :show-inheritance:

Cluster
------------------------------

.. automodule:: relevanceai.api.cluster
   :members:
   :undoc-members:
   :show-inheritance:

Datasets
-------------------------------

.. automodule:: relevanceai.api.datasets
   :members:
   :undoc-members:
   :show-inheritance:

Document Operations
--------------------------------

.. automodule:: relevanceai.api.documents
   :members:
   :undoc-members:
   :show-inheritance:

Available Encoders
-------------------------------

.. automodule:: relevanceai.api.encoders
   :members:
   :undoc-members:
   :show-inheritance:

Monitoring
------------------------------

.. automodule:: relevanceai.api.monitor
   :members:
   :undoc-members:
   :show-inheritance:

Recommendations
--------------------------------

.. automodule:: relevanceai.api.recommend
   :members:
   :undoc-members:
   :show-inheritance:

Request Configuration
---------------------------------------

.. automodule:: relevanceai.api.requests_config
   :members:
   :undoc-members:
   :show-inheritance:

Search
-----------------------------

.. automodule:: relevanceai.api.search
   :members:
   :undoc-members:
   :show-inheritance:

Tasks
----------------------------

.. automodule:: relevanceai.api.tasks
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: relevanceai.api
   :members:
   :undoc-members:
   :show-inheritance:
