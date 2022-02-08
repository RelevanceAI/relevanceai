import pytest

import uuid

import numpy as np


@pytest.fixture(scope="session", autouse=True)
def sample_numpy_documents():
    def _sample_numpy_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1_numpy": np.random.randint(5, size=1)[0],
            "sample_2_numpy": np.random.rand(3, 2),
            "sample_3_numpy": np.nan,
        }

    N = 20
    return [_sample_numpy_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]
