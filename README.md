# RelevanceAI

For guides, tutorials on how to use this package, visit https://docs.relevance.ai/docs.

If you are looking for an SDK reference, you can find that [here](https://youthful-leakey-ab1977.netlify.app/index.html).

Built mainly for users looking to experiment with vectors/embeddings.

## Installation 

The easiest way is to install this package is to run `pip install --upgrade relevanceai`.

## How to use the RelevanceAI client

For the relevanceai client, we want to ensure the SDK mirrors the API client.

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

## Features

### Out of the box multi-threading/multi-processing

Get multi-threading and multi-processing out of the box. The relevanceai Python package automatically gives you multi-threading and multi-processing out of the box!

```python

client.insert_documents(
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

client.insert_documents(
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
client.pull_update_push(original_collection, even_function, new_collection)
```

An example of the data now: 
```
[{"_id": "0", "even": true}, {"_id": "1", "even": false}, ... {"_id": "199", "even": true}]
```

To delete all logs created by pull_update_push, use the `delete_all_logs` function.
```python
client.delete_all_logs(original_collection)
```

### Sample Datasets 

If you require a sample dataset, you can run the following to help:


```python
from relevanceai.datasets import get_ecommerce_dataset
docs = get_ecommerce_dataset()
```


### Integration with VectorHub

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
    return model.encode_documents(
        ["current", "Longer"], docs)

client.pull_update_push("sample-cn", encode_documents)

```
### Visualisation

To use the Embedding projector - 

See [`relevanceai/visualise/constants.py`]('./relevanceai/visualise/constants.py') for default args . avaialble


```python
from relevanceai import Client


client = Client(project=project, api_key=api_key, base_url=base_url)

'''
Retrieve docs in dataset  set `number_of_points_to_render = None` to retrieve all docs
'''

vector_label = "product_name"
vector_field = "product_name_imagetext_vector_"

dr = 'pca'
cluster = 'kmeans'

client.projector.plot(
    dataset_id="ecommerce-6", 
    vector_field=vector_field,
    number_of_points_to_render=1000,
)  
```


Full options and more details on functionality, see [this notebook](https://colab.research.google.com/drive/1ONEjcIf1CqUhXy8dknlyAnp1DnSAYHnR?usp=sharing) here - 


```python

'''
If `cluster` specified, will override `colour_label` option and render cluster as legend
'''

dr = 'tsne'
cluster = 'kmedoids'

dataset_id = "ecommerce-6"
vector_label = "product_name"
vector_field = "product_name_imagetext_vector_"

client.projector.plot(
    dataset_id = dataset_id,
    vector_field = vector_field,
    number_of_points_to_render=1000,
    
    ### Dimensionality reduction args
    dr = dr,
    dr_args = DIM_REDUCTION_DEFAULT_ARGS[ dr ], 

    ## Plot rendering args
    vector_label = None, 
    colour_label = vector_label,
    hover_label = None,
    
    ### Cluster args
    cluster = cluster,
    cluster_args = CLUSTER_DEFAULT_ARGS[ cluster ],
    num_clusters = 20
)

```

## Troubleshooting 


### Pull Update Push

Sometimes Pull Update Push will fail for strange reasons (create a Github Issue!)

If this is the case, then you are free to use this: 

```python
from relevanceai import Client
url = "https://api-aueast.relevance.ai/v1/"

client = Client()
docs = client.datasets.documents.get_where(collection, select_fields=['title'])
while len(docs['documents']) > 0:
    docs['documents'] = model.encode_documents_in_bulk(['product_name'], docs['documents'])
    client.update_documents(collection, docs['documents'])
    docs = client.datasets.documents.get_where(collection, select_fields=['product_name'], cursor=docs['cursor'])
```

### Stop logging 

In order to stop all logging, you can just run this: 

```
client.logger.disable(name="relevanceai")
```

This can be helpful during client demos when you do not need to show the API endpoint being hit.

## Development

### Getting Started

To get started with development, ensure you have `pytest` and `mypy` installed. These will help ensure typechecking and testing.

```
python -m pip install pytest mypy
```
Then run testing using:


```
python -m pytest
mypy relevanceai
```

Otherwise, you can also setup your dev env using the [`Makefile`](./Makefile)

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

To add new targets, add new command and intended script, add descriptive comment above the command block as description to the rule.

```make
#################################################################################
# COMMANDS                                                                      #
#################################################################################
...
## Test dependencies
test
	pytest $(TEST_PATH) --cov=relevanceai -vv -rs
...
```
Then run testing using:

You can limit your testing on a single file/folder by specifying a test path to folder or file.


```zsh

❯ make test TEST_PATH=tests/integration    


pytest . --cov=relevanceai -vv -rs
========================================= test session starts =========================================
platform linux -- Python 3.7.0, pytest-6.2.5, py-1.10.0, pluggy-1.0.0 -- /home/charlene/code/relevanceai/.venv/bin/python3.8
cachedir: .pytest_cache
rootdir: /home/charlene/code/relevanceai
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
relevanceai/__init__.py                  2      0   100%
relevanceai/api/__init__.py              0      0   100%
relevanceai/api/admin.py                 4      4     0%
relevanceai/api/aggregate.py             4      1    75%
relevanceai/api/centroids.py            10      2    80%
relevanceai/api/client.py               12      0   100%
relevanceai/api/cluster.py              10      1    90%
relevanceai/api/datasets.py             60     33    45%
relevanceai/api/documents.py            39     24    38%
relevanceai/api/encoders.py             12      3    75%
relevanceai/api/monitor.py              11      2    82%
relevanceai/api/recommend.py             9      1    89%
relevanceai/api/requests_config.py       0      0   100%
relevanceai/api/search.py               20      9    55%
relevanceai/api/services.py             17      0   100%
relevanceai/api/tasks.py                56     38    32%
relevanceai/base.py                     16      1    94%
relevanceai/batch/__init__.py            0      0   100%
relevanceai/batch/batch_insert.py      127    106    17%
relevanceai/batch/chunk.py              11      5    55%
relevanceai/batch/client.py              9      2    78%
relevanceai/concurrency.py              34     27    21%
relevanceai/config.py                   14      2    86%
relevanceai/datasets.py                 66     16    76%
relevanceai/errors.py                    2      0   100%
relevanceai/http_client.py              26      4    85%
relevanceai/progress_bar.py             54     40    26%
relevanceai/transport.py                59     25    58%
relevanceai/vecdb_logging.py            18     18     0%
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
