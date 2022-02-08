import pytest

import numpy as np

from typing import Dict


@pytest.fixture(scope="session")
def error_document():
    return {"_id": 3, "value": np.nan}
