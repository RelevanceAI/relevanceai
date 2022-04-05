---
title: "Exact match"
slug: "exact_match"
hidden: false
createdAt: "2021-11-25T05:20:53.996Z"
updatedAt: "2022-01-19T05:16:30.996Z"
---
<figure>
<img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/exact-match.png?raw=true" width="2062" alt="Exact match.png" />
<figcaption>Filtering documents with "Durian Leather 2 Seater Sofa" as the product_name.</figcaption>
<figure>

## `exact_match`
This filter works with string values and only returns documents with a field value that exactly matches the filtered criteria. For instance under filtering by 'Samsung galaxy s21', the result will only contain products explicitly having 'Samsung galaxy s21' in their specified field. *Note that this filter is case-sensitive.*

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
        "field": product_name,
        "filter_type": exact_match,
        "condition": ==,
        "condition_value": Durian Leather 2 Seater Sofa
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



