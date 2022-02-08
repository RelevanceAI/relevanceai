import pytest

from typing import Dict, List

from tests.globals.utils import NUMBER_OF_DOCUMENTS


@pytest.fixture(scope="session")
def vector_documents(vector_document: Dict) -> List:
    return [vector_document for _ in range(NUMBER_OF_DOCUMENTS)]
