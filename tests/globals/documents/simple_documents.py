import pytest

from typing import Dict, List

from tests.globals.utils import NUMBER_OF_DOCUMENTS


@pytest.fixture(scope="session")
def simple_documents(simple_document: Dict) -> List:
    return [simple_document for _ in range(NUMBER_OF_DOCUMENTS)]
