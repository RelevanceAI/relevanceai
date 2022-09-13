![Github Banner](assets/github_banner.png)

## Relevance AI - The ML Platform for Unstructured Data Analysis 
[![Documentation Status](https://readthedocs.org/projects/relevanceai/badge/?version=latest)](https://relevanceai.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/relevanceai)](https://img.shields.io/pypi/l/relevanceai)

🌎 80% of data in the world is unstructured in the form of text, image, audio, videos, and more.

🔥 Use Relevance to unlock the value of your unstructured data:
- ⚡ Quickly analyze unstructured data with pre-trained machine learning models in a few lines of code.
- ✨ Visualize your unstructured data. Text highlights from Named entity recognition, Word cloud from keywords, Bounding box from images.
- 📊 Create charts for both structured and unstructured.
- 🔎 Drilldown with filters and similarity search to explore and find insights.
- 🚀 Share data apps with your team.

[Sign up for a free account ->](https://hubs.ly/Q017CkXK0)

Relevance AI also acts as a platform for:
- 🔑 Vectors, storing and querying vectors with flexible vector similarity search, that can be combined with multiple vectors, aggregates and filters.
- 🔮 ML Dataset Evaluation, for debugging dataset labels, model outputs and surfacing edge cases.


## 🧠 Documentation

| Type      | Link |
| ------------- | ----------- |
| Python API | [Documentation](https://sdk.relevance.ai/) |
| Python Reference | [Documentation](https://relevanceai.readthedocs.io/en/latest/)        |
| Cloud Dashboard | [Documentation](https://docs.relevance.ai/) |

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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RelevanceAI/RelevanceAI/blob/development/guides/quickstart_guide.ipynb)

Login to `relevanceai`:
```{python}
from relevanceai import Client

client = Client()
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

```{python}
ds = client.Dataset("quickstart")
ds.insert_documents(docs)
```
> Quick tip! Our Dataset object is compatible with common dataframes methods like `.head()`, `.shape()` and `.info()`.

### Perform vector search

```{python}
query = [
    {"vector": [0.2, 0.2, 0.2], "field": "example_vector_"}
]
results = ds.search(
    vector_search_query=query,
    page_size=3,
)
```
[Learn more about how to flexibly configure your vector search ->](https://sdk.relevance.ai/docs/search)

### Perform clustering

Generate clusters
```{python}
clusterop = ds.cluster(vector_fields=["example_vector_"])
clusterop.list_closest()
```

Generate clusters with sklearn
```{python}
from sklearn.cluster import AgglomerativeClustering

cluster_model = AgglomerativeClustering()
clusterop = ds.cluster(vector_fields=["example_vector_"], model=cluster_model, alias="agglomerative")
clusterop.list_closest()
```
[Learn more about how to flexibly configure your clustering ->](https://sdk.relevance.ai/docs/search)

## 🧰 Config

The config object contains the adjustable global settings for the SDK. For a description of all the settings, see [here](https://github.com/RelevanceAI/RelevanceAI/blob/development/relevanceai/constants/config.ini).

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
