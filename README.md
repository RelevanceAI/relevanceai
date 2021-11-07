# relevanceai Python SDK 

For documentation about how to use this package, visit: https://docs.relevance.ai/docs

This is the Python SDK of relevanceai and is the main Pythonic interface for the relevanceai.

Built mainly for users looking to experiment with vectors/embeddings without having to consistently rely on the `requests` module.

## Installation 

The easiest way is to install this package is to run `pip install --upgrade relevanceai`.

## How to use the relevanceai client

For the relevanceai client, we want to ensure the SDK mirrors the API client.

For example:

```python
## To instantiate the client 
from relevanceai import Client
project = input("Your project goes here")
api_key = input("Your API key goes here")
client = Client(project, api_key)
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

## Features

### Out of the box multi-threading/multi-processing

Get multi-threading and multi-processing out of the box. The relevanceai Python package automatically gives you multi-threading and multi-processing out of the box!

```python

vdb_client.insert_documents(
    dataset_id="match-example",
    docs=participant_frames,
    update_schema=True,
    overwrite=True,
)

def bulk_fn(docs):
    # bulk_fn receives a list of documents (python dictionaries)
    for d in docs:
        d["value_update"] = d["value"] + 2
    return docs

vdb_client.insert_documents(
    dataset_id="match-example",
    docs=participant_frames,
    update_schema=True,
    overwrite=True,
    bulk_fn=bulk_fn
)
```

Under the hood, we use multiprocessing for processing the `bulk_fn` and multi-threading to send data via network requests. However, if there is no `bulk_fn` supplied, it only multi-threads network requests.

### Pull Update Push

Update documents within your collection based on a rule customised by you. The Pull-Update-Push Function loops through every document in your collection, brings it to your local computer where a function is applied (specified by you) and reuploaded to either an new collection or updated in the same collection. There is a logging functionality to keep track of which documents have been updated to save on network requests.

For example, consider a scenario where you have uploaded a dataset called 'test_dataset' containing integers up to 200. 

```python
original_collection = 'test_dataset'
data = [{'_id': str(i)} for i in range(200)]
client.datasets.bulk_insert(original_collection, data)
```

An example of sample data looks like this:

```
[{"_id": "0"}, {"_id": "1"}, ... {"_id": "199"}]
```


Now, you want to create a new field in this dataset to flag whether the number is even or not. This function must input a list of documents (the original collection) and output another list of documents (to be updated).

```python
def even_function(data):
    for i in data:
        if int(i['_id']) % 2 == 0:
            i['even'] = True
        else:
            i['even'] = False
    return data
```
This function is then included in the Pull-Update-Push Function to update every document in the uploaded collection.
 
```python
client.pull_update_push(original_collection, even_function)
```

An example of the data now: 
```
[{"_id": "0", "even": true}, {"_id": "1", "even": false}, ... {"_id": "199", "even": true}]
```

Alternatively, a new collection could be specified to direct where updated documents are uploaded into.

```python
new_collection = 'updated_test_dataset'
vec_client.pull_update_push(original_collection, even_function, new_collection)
```

An example of the data now: 
```
[{"_id": "0", "even": true}, {"_id": "1", "even": false}, ... {"_id": "199", "even": true}]
```

To delete all logs created by pull_update_push, use the `delete_all_logs` function.
```python
vec_client.delete_all_logs(original_collection)
```


## Integration with VectorHub

VectorHub is RelevanceAI's main encoder repository. For the models used here, we have abstracted away a lot of 
complexity from installation to encoding and have innate VectorAI support. 

Using VectorHub models is as simple as (actual example): 

```python
# Inserting a dataframe
import pandas as pd
df = pd.read_csv("Grid view.csv")
df['_id'] = df['sample']
client.insert_df("sample-cn", df)

# !pip install vectorhub[encoders-text-sentence-transformers]
from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec
model = SentenceTransformer2Vec()

# Define an update function
def encode_documents(docs):
    # Field and then the docs go here
    return model.encode_documents(["current", "Longer"], docs)

client.pull_update_push("sample-cn", encode_documents)

```

### Troubleshooting Pull Update Push

Sometimes Pull Update Push will fail for strange reasons (create a Github Issue!)

If this is the case, then you are free to use this: 

```python
from relevanceai import Client
url = "https://api-aueast.relevance.ai/v1/"

collection = ""
project = ""
api_key = ""
client = Client(project, api_key)
docs = client.datasets.documents.get_where(collection, select_fields=['title'])
while len(docs['documents']) > 0:
    docs['documents'] = model.encode_documents_in_bulk(['product_name'], docs['documents'])
    client.update_documents(collection, docs['documents'])
    docs = client.datasets.documents.get_where(collection, select_fields=['product_name'], cursor=docs['cursor'])
```

## Stop logging 

In order to stop all logging, you can just run this: 

```
client.logger.stop()
```

This can be helpful during client demos when you do not need to show the API endpoint being hit.

## Sample Datasets 

If you require a sample dataset, you can run the following to help:


```python
from relevanceai.datasets import get_ecommerce_dataset
docs = get_ecommerce_dataset()
```



## Development


## Development

### Getting Started

Setup has been simplified in [`Makefile`](./Makefile)
Setup your virtualenv, install requirements and package

```python
❯ make install
```

For all available targets

```python
❯ make help
Available rules:

clean               Delete all compiled Python files 
install             Install dependencies 
lint                Lint using flake8 
test                Test dependencies 
update              Update dependencies 
```

To add new targets, add new command and intended script, add descriptive comment above the command block as description to the rule.`

```make
#################################################################################
# COMMANDS                                                                      #
#################################################################################
...
## Test dependencies
test
	pytest $(TEST_PATH) --cov=vecdb -vv
...
```


Run your local tests in [`tests`](./tests)

```zsh
❯ make test
```

You should see similar to below -

```
❯ make test
pytest . --cov=vecdb -vv
========================================= test session starts =========================================
platform linux -- Python 3.8.0, pytest-6.2.5, py-1.10.0, pluggy-1.0.0 -- /home/charlene/code/vecdb/.venv/bin/python3.8
cachedir: .pytest_cache
rootdir: /home/charlene/code/vecdb
plugins: cov-3.0.0, mock-3.6.1, dotenv-0.5.2
collected 12 items

tests/test_datasets.py::test_get_games_dataset_subset PASSED                                    [  8%]
tests/test_datasets.py::test_get_games_dataset_full SKIPPED (Skipping full data load to min...) [ 16%]
tests/test_datasets.py::test_get_online_retail_dataset_subset PASSED                            [ 25%]
tests/test_datasets.py::test_get_online_retail_dataset_full SKIPPED (Skipping full data loa...) [ 33%]
tests/test_datasets.py::test_get_get_news_dataset_subset PASSED                                 [ 41%]
tests/test_datasets.py::test_get_get_news_dataset_full SKIPPED (Skipping full data load to ...) [ 50%]
tests/test_datasets.py::test_get_ecommerce_dataset_subset PASSED                                [ 58%]
tests/test_datasets.py::test_get_ecommerce_dataset_full SKIPPED (Skipping full data load to...) [ 66%]
tests/test_datasets.py::test_get_dummy_ecommerce_dataset_subset PASSED                          [ 75%]
tests/test_datasets.py::test_get_dummy_ecommerce_dataset_full SKIPPED (Skipping full data l...) [ 83%]
tests/test_smoke.py::test_smoke_installation PASSED                                             [ 91%]
tests/test_smoke.py::test_datasets_smoke PASSED                                                 [100%]

----------- coverage: platform linux, python 3.8.0-final-0 -----------
Name                           Stmts   Miss  Cover
--------------------------------------------------
vecdb/__init__.py                  2      0   100%
vecdb/api/__init__.py              0      0   100%
vecdb/api/admin.py                 4      4     0%
vecdb/api/aggregate.py             4      1    75%
vecdb/api/centroids.py            10      2    80%
vecdb/api/client.py               12      0   100%
vecdb/api/cluster.py              10      1    90%
vecdb/api/datasets.py             60     33    45%
vecdb/api/documents.py            39     24    38%
vecdb/api/encoders.py             12      3    75%
vecdb/api/monitor.py              11      2    82%
vecdb/api/recommend.py             9      1    89%
vecdb/api/requests_config.py       0      0   100%
vecdb/api/search.py               20      9    55%
vecdb/api/services.py             17      0   100%
vecdb/api/tasks.py                56     38    32%
vecdb/base.py                     16      1    94%
vecdb/batch/__init__.py            0      0   100%
vecdb/batch/batch_insert.py      127    106    17%
vecdb/batch/chunk.py              11      5    55%
vecdb/batch/client.py              9      2    78%
vecdb/concurrency.py              34     27    21%
vecdb/config.py                   14      2    86%
vecdb/datasets.py                 66     16    76%
vecdb/errors.py                    2      0   100%
vecdb/http_client.py              26      4    85%
vecdb/progress_bar.py             54     40    26%
vecdb/transport.py                59     25    58%
vecdb/vecdb_logging.py            18     18     0%
--------------------------------------------------
TOTAL                            702    364    48%


==================================== 7 passed, 5 skipped in 13.52s ====================================

```


```
Copyright (C) Relevance AI - All Rights Reserved
Unauthorized copying of this repository, via any medium is strictly prohibited
Proprietary and confidential
Relevance AI <dev@relevance.ai> 2021 
```
