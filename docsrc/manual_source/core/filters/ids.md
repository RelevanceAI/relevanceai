---
title: "Ids"
slug: "ids"
hidden: false
createdAt: "2021-11-25T22:22:07.285Z"
updatedAt: "2022-01-19T05:17:10.638Z"
---
<figure>
<img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/id.png?raw=true" width="612" alt="id.png" />
<figcaption>Filtering documents based on their id.</figcaption>
<figure>

## `ids`
This filter returns documents whose unique id exists in a given list. It may look similar to 'categories'. The main difference is the search speed.

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
        "field": _id,
        "filter_type": ids,
        "condition": ==,
        "condition_value": 7790e058cbe1b1e10e20cd22a1e53d36
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

