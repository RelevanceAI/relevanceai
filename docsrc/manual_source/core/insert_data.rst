Inserting Data 
==================

Inserting CSVs
-------------------

.. code-block::

    from relevanceai import Client
    client = Client()
    df = client.Dataset("sample_dataset_id")

    csv_filename = "temp.csv"
    df.insert_csv(csv_filename)

Inserting Pandas Dataframes
-------------------------------

Insert a dataframe into the dataset. Takes additional args and kwargs based on insert_documents.

.. code-block::

    from relevanceai import Client
    client = Client()
    df = client.Dataset("sample_dataset_id")
    pandas_df = pd.DataFrame({"value": [3, 2, 1], "_id": ["10", "11", "12"]})
    df.insert_pandas_dataframe(pandas_df)

Inserting Media
------------------

Given a path to a directory, this method loads all media-related files into a Dataset.

.. code-block::

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("dataset_id")

    from pathlib import Path
    path = Path("medias/")
    # list(path.iterdir()) returns
    # [
    #    PosixPath('media.jpg'),
    #    PosixPath('more-medias'), # a directory
    # ]

    get_all_medias: bool = True
    if get_all_medias:
        # Inserts all medias, even those in the more-medias directory
        ds.insert_media_folder(
            field="medias", path=path, recurse=True
        )
    else:
        # Only inserts media.jpg
        ds.insert_media_folder(
            field="medias", path=path, recurse=False
        )

Inserting Documents (JSON-like objects)
-----------------------------------------

.. code-block::

    from relevanceai import Client

    client = Client()

    dataset_id = "sample_dataset_id"
    df = client.Dataset(dataset_id)

    documents = [
        {
            "_id": "10",
            "value": 5
        },
        {
            "_id": "332",
            "value": 10
        }
    ]

    df.insert_documents(documents)

Insert a list of documents with multi-threading automatically enabled.

When inserting the document you can optionally specify your own id for a document by using the field name “_id”, if not specified a random id is assigned.

When inserting or specifying vectors in a document use the suffix (ends with) “_vector_” for the field name. e.g. “product_description_vector_”.

When inserting or specifying chunks in a document the suffix (ends with) “_chunk_” for the field name. e.g. “products_chunk_”.

When inserting or specifying chunk vectors in a document’s chunks use the suffix (ends with) “_chunkvector_” for the field name. e.g. “products_chunk_.product_description_chunkvector_”.


