import pytest

import uuid

from typing import List

from tests.globals.constants import NUMBER_OF_DOCUMENTS

from tests.globals.document import numpy_document


@pytest.fixture(scope="session")
def numpy_documents() -> List:
    return [numpy_document(uuid.uuid4().__str__()) for _ in range(NUMBER_OF_DOCUMENTS)]
