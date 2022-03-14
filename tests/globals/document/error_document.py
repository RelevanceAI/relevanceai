import numpy as np

from relevanceai.package_utils.utils import _make_id


def error_document():
    document = {"value": np.nan}
    document["_id"] = _make_id(document)
    return document
