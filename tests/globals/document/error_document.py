import numpy as np

from relevanceai.dataset.crud.helpers import make_id


def error_document():
    document = {"value": np.nan}
    document["_id"] = make_id(document)
    return document
