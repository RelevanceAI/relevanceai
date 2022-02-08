import pytest

import uuid

import numpy as np
import pandas as pd


@pytest.fixture(scope="session")
def pandas_document():
    return {
        "_id": uuid.uuid4().__str__(),
        "sample_1_pandas": pd.DataFrame(
            np.random.randint(0, 20, size=(20, 4)), columns=list("ABCD")
        ),
        "sample_2_pandas": pd.DataFrame(
            np.random.randint(0, 10, size=(10, 4)), columns=list("ABCD")
        ),
        "sample_3_pandas": pd.DataFrame(
            np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]]),
            columns=["a", "b", "c"],
        ),
    }
