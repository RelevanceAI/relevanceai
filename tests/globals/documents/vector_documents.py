import pytest

import uuid

from typing import List

from tests.globals.constants import NUMBER_OF_DOCUMENTS

from tests.globals.document import vector_document


@pytest.fixture(scope="session")
def vector_documents() -> List:
    return [vector_document(uuid.uuid4().__str__()) for _ in range(NUMBER_OF_DOCUMENTS)]
