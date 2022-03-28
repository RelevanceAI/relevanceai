import numpy as np

from relevanceai.utils import make_id


def error_document():
    document = {"value": np.nan}
    document["_id"] = make_id(document)
    return document
