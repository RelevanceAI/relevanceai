![Github Banner](assets/github_banner.png)

[![Documentation Status](https://readthedocs.org/projects/relevanceai/badge/?version=latest)](https://relevanceai.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/relevanceai)](https://img.shields.io/pypi/l/relevanceai)

[Join our slack channel!](https://join.slack.com/t/relevance-ai/shared_invite/zt-11fo8oush-dHPd57wamhoQ7J5arNv1mg)

**Run Our Colab Notebook And Get Started In Less Than 10 Lines Of Code!**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://hubs.ly/Q014Qbs10)

For guides and tutorials on how to use this package, visit https://docs.relevance.ai/docs.

## 🔥 Features

- Fast vector search with free dashboard to preview and visualise results
- Vector clustering with support for libraries like scikit-learn and easy built-in customisation
- Store nested documents with support for multiple vectors and metadata in one object
- Multi-vector search with filtering, facets, weighting
- Hybrid search with support for weighting keyword matching and vector search
... and more!


## 🧠 Documentation

| API type      | Link |
| ------------- | ----------- |
| Guides | [Documentation](https://docs.relevance.ai/) |
| Python Reference | [Documentation](https://relevanceai.readthedocs.io/en/latest/)        |

You can easily access our documentation while using the SDK using:

```{python}
from relevanceai import Client
client = Client()

# Easy one line of code to access our docs
client.docs

```


## 🛠️ Installation

Using pip:

```{bash}
pip install -U relevanceai
```
Using conda:

```{bash}
conda install -c relevance relevanceai
```

## ⏩ Quickstart

### Login into your project space

```{python}
from relevanceai import Client

client = Client(<project_name>, <api_key>)
```

Prepare your documents for insertion by following the below format:
- Each document should be a dictionary
- Include a field `_id` as a primary key, otherwise it's automatically generated
- Suffix vector fields with `_vector_`

```{python}
docs = [
    {"_id": "1", "example_vector_": [0.1, 0.1, 0.1], "data": "Documentation"},
    {"_id": "2", "example_vector_": [0.2, 0.2, 0.2], "data": "Best document!"},
    {"_id": "3", "example_vector_": [0.3, 0.3, 0.3], "data": "document example"},
    {"_id": "4", "example_vector_": [0.4, 0.4, 0.4], "data": "this is another doc"},
    {"_id": "5", "example_vector_": [0.5, 0.5, 0.5], "data": "this is a doc"},
]
```

### Insert data into a dataset

Create a dataset object with the name of the dataset you'd like to use. If it doesn't exist, it'll be created for you.
> Quick tip! Our Dataset object is compatible with common dataframes methods like `.head()`, `.shape()` and `.info()`.

```{python}
ds = client.Dataset("quickstart")
ds.insert_documents(docs)
```

### Perform vector search

```{python}
results = ds.vector_search(
    multivector_query=[{"vector": [0.2, 0.2, 0.2], "fields": ["example_vector_"]}],
    page_size=3,
    query="sample search" # optional, name to display in dashboard
)
```

### Cluster dataset with Auto Cluster

Generate 12 clusters using kmeans
```{python}
clusterop = ds.auto_cluster("kmeans-12", vector_fields=["example_vector_"])
clusterop.list_closest_to_center()
```
> Quick tip! After each of these steps, the output will provide a URL to the Relevance AI dashboard where you can see a visualisation of your results

## 🚧 Development

### Getting Started
To get started with development, ensure you have pytest and mypy installed. These will help ensure typechecking and testing.

```{bash}
python -m pip install pytest mypy
```

Then run testing using:

> Don't forget to set your test credentials!

```{bash}
export TEST_PROJECT = xxx
export TEST_API_KEY = xxx

python -m pytest
mypy relevanceai
```

Set up precommit

```{bash}
pip install precommit
pre-commit install
```

## 🧰 Config

The config object contains the adjustable global settings for the SDK. For a description of all the settings, see here.

To view setting options, run the following:

```{python}
client.config.options
```

The syntax for selecting an option is section.key. For example, to disable logging, run the following to modify logging.enable_logging:

```{python}
client.config.set_option('logging.enable_logging', False)
```

To restore all options to their default, run the following:

### Changing the base URL

You can change the base URL as such:

```{python}
client.base_url = "https://.../latest"
```

You can also update the ingest base URL:

```{python}
client.ingest_base_url = "https://.../latest
```
