---
title: "Regex"
slug: "regex"
hidden: false
createdAt: "2021-11-29T23:13:52.305Z"
updatedAt: "2022-01-19T05:16:17.784Z"
---
<figure>
<img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/regex.png?raw=true" width="2048" alt="7cbd106-contains.png" />
<figcaption>Filtering documents containing "Durian (\w+)" in description using filter_type `regexp`.</figcaption>
<figure>

## `regex`
This filter returns a document only if it matches regexp (i.e. regular expression). Note that substrings are covered in this category. For instance, if a product name is composed of a name and a number (e.g. ABC-123), one might remember the name but not the number. This filter can easily return all products including the ABC string.

Relevance AI has the same regular expression schema as Apache Lucene's ElasticSearch to parse queries.

*Note that this filter is case-sensitive.*

```bash Bash
# remove `!` if running the line in a terminal
!pip install -U RelevanceAI[notebook]==1.4.3
```
```bash
```

```python Python (SDK)
from relevanceai import Client

"""
You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
Once you have signed up, click on the value under `Activation token` and paste it here
"""
client = Client()
```
```python
```

```python Python (SDK)
DATASET_ID = "ecommerce-sample-dataset"
df = client.Dataset(DATASET_ID)
```
```python
```

```python Python (SDK)
filter = [
    {
        "field": description,
        "filter_type": regexp,
        "condition": ==,
        "condition_value": .*Durian (\w+)
    }
]
```
```python
```

```python Python (SDK)
### TODO: update to match the latest SDK
filtered_data = df.get_where(filter)
```
```python
```


