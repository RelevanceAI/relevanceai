# VecDB Python SDK 

This is the Python SDK of VecDB. This is the main Pythonic interface for the VecDB client.
Built mainly for users looking to experiment with vectors/embeddings without having to consistently
rely on the API.

## Installation 

The easiest way is to clone this package to your local directory, navigate to the directory and then run `pip install vecdb`.

This will automatically let you unlock the power of VecDB without having to rely on re-constructing API requests.

## How to use this package 

Instantiating the client.

```{python}
from vecdb import VecDBClient
client = VecDBClient(project, api_key)
```

Get multi-threading and multi-processing out of the box. The VecDB Python package automatically gives you multi-threading and multi-processing out of the box!

```{python}

def bulk_fn(docs):
    # bulk_fn receives a list of documents
    for d in docs:
        d["value_update"] = d["value"] + 2
    return docs

vdb_client.insert_documents(
    dataset_id="match-example",
    docs=participant_frames[0:50],
    update_schema=True,
    overwrite=True,
    bulk_fn=bulk_fn
)
```

Under the hood, our engineering team uses multiprocessing for processing the `bulk_fn` and then automatically uses multi-threading to send data via network requests. However, if there is no `bulk_fn` supplied, it automatically multi-threads network requests.
