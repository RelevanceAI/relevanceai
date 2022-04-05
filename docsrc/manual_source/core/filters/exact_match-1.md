---
title: "Word match"
slug: "exact_match-1"
hidden: false
createdAt: "2021-11-25T05:44:25.366Z"
updatedAt: "2022-01-19T05:16:37.437Z"
---
<figure>
<img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/word-match.png?raw=true" width="1974" alt="wordmatch.png" />
<figcaption>Filtering documents matching "Home curtain" in the description field.</figcaption>
<figure>

## `word_match`
This filter has similarities to both `exact_match` and `contains`. It returns a document only if it contains a **word** value matching the filter; meaning substrings are covered in this category but as long as they can be extracted with common word separators like the white-space (blank). For instance, the filter value "Home Gallery",  can lead to extraction of a document with "Buy Home Fashion Gallery Polyester ..." in the description field as both words are explicitly seen in the text. *Note that this filter is case-sensitive.*

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
        "filter_type": word_match,
        "condition": ==,
        "condition_value": Home curtain
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

