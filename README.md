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

## Development

### Getting Started

To get started with development, ensure you have `pytest` and `mypy` installed. These will help ensure typechecking and testing.

```python
python -m pip install pytest mypy
```


Then run testing using:

Make sure to set your test credentials!

```
export TEST_PROJECT = xxx 
export TEST_API_KEY = xxx 
```

```python
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
	pytest $(TEST_PATH) --cov=relevanceai -vv -rs -x
...
```
Then run testing using:

You can limit your testing on a single file/folder by specifying a test path to folder or file.


```zsh

❯ make test TEST_PATH=tests/integration


pytest . --cov=relevanceai -vv -rs -x
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
