import pytest

import uuid

import numpy as np


@pytest.fixture(scope="session")
def numpy_document():
    return {
        "_id": uuid.uuid4().__str__(),
        "sample_1_numpy": np.random.randint(5, size=1)[0],
        "sample_2_numpy": np.random.rand(3, 2),
        "sample_3_numpy": np.nan,
    }
