import pytest

from typing import Dict, List

from tests.globals.utils import NUMBER_OF_DOCUMENTS


@pytest.fixture(scope="session")
def assorted_nested_documents(complex_nested_document: Dict) -> List:
    return [complex_nested_document for _ in range(NUMBER_OF_DOCUMENTS)]


@pytest.fixture(scope="session")
def simple_nested_docs(simple_nested_document: Dict) -> List:
    return [simple_nested_document for _ in range(NUMBER_OF_DOCUMENTS)]
