import numpy as np


def do_nothing(documents):
    return documents


def add_new_column(documents, column, value):
    for document in documents:
        document[column] = value
    return documents


def cause_error(documents):
    for d in documents:
        d["value"] = np.nan
    return documents


def cause_some_error(documents):
    MAX_ERRORS = 5
    ERROR_COUNT = 0
    for d in documents:
        if ERROR_COUNT < MAX_ERRORS:
            d["value"] = np.nan
            ERROR_COUNT += 1
    return documents
