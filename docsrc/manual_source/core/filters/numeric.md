---
title: "Numeric"
slug: "numeric"
hidden: false
createdAt: "2021-11-25T06:28:37.534Z"
updatedAt: "2022-01-19T05:17:05.147Z"
---
<figure>
<img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/numeric.png?raw=true" width="446" alt="Numeric.png" />
<figcaption>Filtering documents with retail price higher than 5000.</figcaption>
<figure>

## `numeric`
This filter is to perform the filtering operators on a numeric value. For instance, returning the documents with a price larger than 1000 dollars.

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
        "field": retail_price,
        "filter_type": numeric,
        "condition": >,
        "condition_value": 5000
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


