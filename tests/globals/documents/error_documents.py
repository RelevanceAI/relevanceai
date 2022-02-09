import pytest

import uuid

from typing import List

from tests.globals.constants import NUMBER_OF_DOCUMENTS

from tests.globals.document import error_document


@pytest.fixture(scope="session")
def error_documents() -> List:
    return [error_document(uuid.uuid4().__str__()) for _ in range(NUMBER_OF_DOCUMENTS)]
