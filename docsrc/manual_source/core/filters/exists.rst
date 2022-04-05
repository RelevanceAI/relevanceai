---
title: "Exists"
slug: "exists"
hidden: false
createdAt: "2021-11-25T06:09:19.375Z"
updatedAt: "2022-01-19T05:16:52.455Z"
---
<figure>
<img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/exists.png?raw=true" width="500" alt="exist.png" />
<figcaption>Filtering documents which include the field "brand" in their information.</figcaption>
<figure>

## `exists`
This filter returns entries in a database if a certain field (as opposed to the field values in previously mentioned filter types) exists or doesn't exist in them. For instance, filtering out documents in which there is no field 'purchase-info'. *Note that this filter is case-sensitive.*

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
        "field": brand,
        "filter_type": exists,
        "condition": ==,
        "condition_value":  
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


