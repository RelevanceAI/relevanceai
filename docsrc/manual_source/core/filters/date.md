---
title: "Date"
slug: "date"
hidden: false
createdAt: "2021-11-25T06:20:15.175Z"
updatedAt: "2022-01-19T05:16:58.720Z"
---
<figure>
<img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/date.png?raw=true" width="600"  alt="date.png" />
<figcaption>Filtering documents which were added to the database after January 2021.</figcaption>
<figure>

## `date`
This filter performs date analysis and filters documents based on their date information. For instance, it is possible to filter out any documents with a production date before January 2021.

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
        "field": "insert_date_",
        "filter_type": date,
        "condition": ==,
        "condition_value": 2020-07-01
    }
]
```
```python
```

Note that the default format is "yyyy-mm-dd" but can be changed to "yyyy-dd-mm" through the `format` parameter as shown in the example below.

```python Python (SDK)
filter = [
    {
        "field": "insert_date_",
        "filter_type": date,
        "condition": ==,
        "condition_value": 2020-07-01,
        "format": "yyyy-dd-MM"
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

