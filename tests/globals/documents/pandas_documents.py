import pytest

from typing import List

from tests.globals.constants import NUMBER_OF_DOCUMENTS

from tests.globals.document import pandas_document


@pytest.fixture(scope="session")
def pandas_documents() -> List:
    return [pandas_document() for _ in range(NUMBER_OF_DOCUMENTS)]
