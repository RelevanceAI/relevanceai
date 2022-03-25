"""
Helper functions for the pandas like dataset api
"""
import uuid

from typing import Union, List


def _create_base_filter(field, filter_type, condition, condition_value):
    return {
        "field": field,
        "filter_type": filter_type,
        "condition": condition,
        "condition_value": condition_value,
    }


def _build_filters(value: Union[List, str], filter_type: str, index: str):
    """
    Given a filter_dict, create a list of json-like filters with filter_type to interact with SDK

    Parameters
    ----------

    filter_dict: dict
        a dictionary of lists or strings to fitler on

    filter_type: str
        the type of filter to create

    Returns
    -------
    filter: list
        A list of dictionary filters to use in a get_where-like query

    Example
    -------

    >>> filter_dict = {"_id": [1, 2, 3], "column1": 4, "column2": ["cat", "dog"]}
    >>> filter_type = "exact_match"
    >>> filters = build_filters(filter_dict, filter_type)
    """
    filters = []
    if isinstance(value, str):
        filter = _create_base_filter(index, filter_type, "==", value)
        filters.append(filter)

    elif isinstance(value, list):
        for subvalue in value:
            filter = _create_base_filter(index, filter_type, "==", subvalue)
            filters.append(filter)

    return filters
