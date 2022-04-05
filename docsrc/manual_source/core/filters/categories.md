---
title: "Categories"
slug: "categories"
hidden: false
createdAt: "2021-11-25T06:03:45.435Z"
updatedAt: "2022-01-19T05:16:45.084Z"
---
<figure>
<img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/category.png?raw=true" width="658" alt="categories.png" />
<figcaption>Filtering documents with "LG" or "Samsung" as the brand.</figcaption>
<figure>

## `categories`

This filter checks the entries in a database and returns ones in which a field value exists in a given filter list. For instance, if the product name is any of Sony, Samsung, or LG. *Note that this filter is case-sensitive.*

```python Python (SDK)
filter = [
    {
        "field": brand,
        "filter_type": categories,
        "condition": >=,
        "condition_value": ['LG', 'samsung']
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

