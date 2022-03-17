import pytest

from typing import List

from tests.globals.constants import NUMBER_OF_DOCUMENTS

from tests.globals.document import complex_nested_document, simple_nested_document


@pytest.fixture(scope="session")
def assorted_nested_documents() -> List:
    return [complex_nested_document() for _ in range(NUMBER_OF_DOCUMENTS)]


@pytest.fixture(scope="session")
def simple_nested_documents() -> List:
    return [simple_nested_document() for _ in range(NUMBER_OF_DOCUMENTS)]
