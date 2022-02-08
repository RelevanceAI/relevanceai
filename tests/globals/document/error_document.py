import pytest

import numpy as np


@pytest.fixture(scope="session", autouse=True)
def error_doc():
    return [{"_id": 3, "value": np.nan}]
