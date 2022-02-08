import pytest

import random

import uuid

import numpy as np
import pandas as pd

from datetime import datetime


@pytest.fixture(scope="session", autouse=True)
def sample_nested_assorted_documents():
    def _sample_nested_assorted_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1": {
                "panda": pd.DataFrame(
                    np.random.randint(0, 20, size=(20, 4)), columns=list("ABCD")
                ),
                "datetime": datetime.now(),
                "numpy": np.random.rand(3, 2),
                "test1": random.random(),
                "test2": random.random(),
            },
            "sample_2": {
                "subsample1": {
                    "panda": pd.DataFrame(
                        np.random.randint(0, 20, size=(20, 4)), columns=list("ABCD")
                    ),
                    "datetime": datetime.now(),
                    "numpy": np.random.rand(3, 2),
                },
                "subsample2": {
                    "panda": pd.DataFrame(
                        np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]]),
                        columns=["a", "b", "c"],
                    ),
                    "datetime": datetime.now(),
                    "numpy": np.random.rand(3, 2),
                },
            },
        }

    N = 20
    return [
        _sample_nested_assorted_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)
    ]


@pytest.fixture(scope="session")
def test_simple_nested_docs():
    def _simple_nested_doc(id):
        return {
            "_id": id,
            "col1": {"subcol1": random.random(), "subcol2": random.random()},
            "col2": {"subcol3": random.random(), "subcol4": random.random()},
            "col3": {"subcol5": random.random(), "subcol6": random.random()},
            "col4": {"subcol7": random.random(), "subcol8": random.random()},
        }

    return [_simple_nested_doc(uuid.uuid4().__str__())] * 20
