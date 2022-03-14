"""
Checks for available integrations in the Python module.
"""

import importlib


def is_faiss_available():
    return importlib.util.find_spec("faiss") is not None


def is_scipy_available():
    return importlib.util.find_spec("scipy") is not None


def is_sklearn_available():
    if importlib.util.find_spec("sklearn") is None:
        return False
    return is_scipy_available() and importlib.util.find_spec("sklearn.metrics")


def is_hdbscan_available():
    return importlib.util.find_spec("hdbscan") is not None
