import pytest

from typing import Dict, List

from tests.globals.utils import NUMBER_OF_DOCUMENTS


@pytest.fixture(scope="session")
def error_documents(error_document: Dict) -> List:
    return [error_document for _ in range(NUMBER_OF_DOCUMENTS)]
