import pytest

from typing import List

from tests.globals.constants import NUMBER_OF_DOCUMENTS

from tests.globals.document import numpy_document


@pytest.fixture(scope="session")
def numpy_documents() -> List:
    return [numpy_document() for _ in range(NUMBER_OF_DOCUMENTS)]
