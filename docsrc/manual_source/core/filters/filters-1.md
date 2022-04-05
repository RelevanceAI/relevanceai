---
title: "Filters"
slug: "filters-1"
hidden: false
createdAt: "2021-11-21T07:05:21.922Z"
updatedAt: "2022-01-19T04:01:08.086Z"
---
<figure>
<img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/filters-1.png?raw=true" width="1009" alt="604547f-combined_filters.png" />
<figcaption>Example output of filtering Lenovo products all inserted into the database after 01/01/2020</figcaption>
<figure>

Filters are great tools to retrieve a subset of documents whose data match the criteria specified in the filter.
For instance, in an e-commerce dataset, we can retrieve all products:
* with prices between 200 and 300 dollars
* with the phrase "free return" included in `description` field
* that are produced after January 2020

> 📘 Filters help us find what we need.
>
> Filters are great tools to retrieve a subset of documents whose data match certain criteria. This allows us to have a more fine-grained overview of the data since only documents that meet the filtering criteria will be displayed.
>
# How to form a filter?

Filters at Relevance AI are defined as Python dictionaries with four main keys:
- `field` (i.e. the data filed in the document you want to filter on)
- `condition` (i.e. operators such as greater than or equal)
- `filter_type` (i.e. the type of filter you want to apply - whether it be date/numeric/text etc.)
- `condition_value` (dependent on the filter type but decides what value to filter on)


```python Python (SDK)
filter = [
    {
        "field": description,
        "filter_type": contains,
        "condition": ==,
        "condition_value": Durian Club
    }
]
```
```python
```

## Filtering operators
Relevance AI covers all common operators:
* "==" (a == b, a equals b)
* "!="  (a != b, a not equals b)
* ">=" (a >= b, a greater that or equals b)
* ">"   (a > b, a greater than b)
* "<"   (a < b, a smaller than b)
* "<=" (a <= b, a smaller than or equals b)

## Filter types
Supported filter types at Relevance AI are listed below.

* contains
* exact_match
* word_match
* categories
* exists
* date
* numeric
* ids
* support for mixing together multiple filters such as in OR situations

We will explain each filter type followed by a sample code snippet in the next pages. There is also a [guide](https://docs.relevance.ai/docs/combining-filters-and-vector-search) on how to combine filters and vector search.

