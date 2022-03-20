import pytest

from typing import List

from tests.globals.constants import NUMBER_OF_DOCUMENTS

from tests.globals.document import datetime_document


@pytest.fixture(scope="session")
def datetime_documents() -> List:
    return [datetime_document() for _ in range(NUMBER_OF_DOCUMENTS)]
