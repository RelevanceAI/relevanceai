Pandas-like Datasets API
================================

In order to make our API more intuitive and easy-to-use for data scientists,
we have switched adopted Pandas' API.

In order to instantiate the Pandas API, you just need to run: 

>>> from relevanceai import Client
>>> client = Client()
>>> dataset_id = "<dataset_id>"
>>> df = client.Dataset(dataset_id)
>>> df.head()

Currently, the main supported Python commands are:
- df.info()
- df.head() # shows preview of Pandas dataframe
- df["field_1"][0] # selects first value in a series
- df.sample() # Randomly 
- df["field_1"].groupby(["value"]).agg({"value_2": "avg"})
- df.apply(lambda x: x + 1) # Apply functionalities are also supported
- df.describe()


Submodules
----------

relevanceai.dataset\_api.centroids module
-----------------------------------------

.. automodule:: relevanceai.dataset_api.centroids
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.dataset\_api.dataset module
---------------------------------------

.. automodule:: relevanceai.dataset_api.dataset
   :members:
   :undoc-members:
   :show-inheritance:

relevanceai.dataset\_api.groupby module
---------------------------------------

.. automodule:: relevanceai.dataset_api.groupby
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: relevanceai.dataset_api
   :members:
   :undoc-members:
   :show-inheritance:
