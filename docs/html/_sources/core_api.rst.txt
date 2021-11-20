Core RelevanceAI Specs
=======================

The core RelevanceAI specs can be found in this section.

For the RelevanceAI client, we want to ensure the SDK mirrors the API client.

For example:

```python
## To instantiate the client 
from relevanceai import Client
client = Client()
```

To use the following endpoint: 

`/datasets/bulk_insert`

You can run: 

```python
# Bulk insert documents
client.datasets.bulk_insert(dataset_id, documents)
```

Or similarly, when you are trying to run 

`/services/search/vector`

You then write: 
```python
# Vector search in a dataset
client.services.search.vector(...)
```

Submodules
----------

relevanceai.api.admin module
----------------------------

.. automodule:: relevanceai.api.admin
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.aggregate module
--------------------------------

.. automodule:: relevanceai.api.aggregate
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.centroids module
--------------------------------

.. automodule:: relevanceai.api.centroids
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.client module
-----------------------------

.. automodule:: relevanceai.api.client
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.cluster module
------------------------------

.. automodule:: relevanceai.api.cluster
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.datasets module
-------------------------------

.. automodule:: relevanceai.api.datasets
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.documents module
--------------------------------

.. automodule:: relevanceai.api.documents
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.encoders module
-------------------------------

.. automodule:: relevanceai.api.encoders
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.monitor module
------------------------------

.. automodule:: relevanceai.api.monitor
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.recommend module
--------------------------------

.. automodule:: relevanceai.api.recommend
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.requests\_config module
---------------------------------------

.. automodule:: relevanceai.api.requests_config
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.search module
-----------------------------

.. automodule:: relevanceai.api.search
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.services module
-------------------------------

.. automodule:: relevanceai.api.services
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.api.tasks module
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
