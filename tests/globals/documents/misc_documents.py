import pytest

from typing import List

from tests.globals.constants import NUMBER_OF_DOCUMENTS

from tests.globals.document import dataclass_document


@pytest.fixture(scope="session")
def dataclass_documents() -> List:
    return [dataclass_document() for _ in range(NUMBER_OF_DOCUMENTS)]
