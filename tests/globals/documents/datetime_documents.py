import pytest

import uuid

from datetime import datetime


@pytest.fixture(scope="session", autouse=True)
def sample_datetime_documents():
    def _sample_datetime_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1_datetime": datetime.now(),
            "sample_2_datetime": datetime.now(),
        }

    N = 20
    return [_sample_datetime_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]
