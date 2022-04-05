---
title: "Contains"
slug: "contains"
hidden: false
createdAt: "2021-11-25T05:18:31.045Z"
updatedAt: "2022-01-19T05:16:24.396Z"
---
<figure>
<img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/contains.png?raw=true" width="2048" alt="contains.png" />
<figcaption>Filtering documents containing "Durian BID" in description using filter_type `contains`.</figcaption>
<figure>


## `contains`

This filter returns a document only if it contains a string value. Note that substrings are covered in this category. For instance, if a product name is composed of a name and a number (e.g. ABC-123), one might remember the name but not the number. This filter can easily return all products including the ABC string.
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
        "filter_type": contains,
        "condition": ==,
        "condition_value": Durian BID
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


