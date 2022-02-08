import pytest

import uuid

from typing import Dict

from datetime import datetime


@pytest.fixture(scope="session")
def datetime_document() -> Dict:
    return {
        "_id": uuid.uuid4().__str__(),
        "sample_1_datetime": datetime.now(),
        "sample_2_datetime": datetime.now(),
    }
