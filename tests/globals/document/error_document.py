import pytest

import numpy as np


@pytest.fixture(scope="session")
def error_document(id: str):
    return {"_id": id, "value": np.nan}
