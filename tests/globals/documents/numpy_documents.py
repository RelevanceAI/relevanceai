import pytest

from typing import Dict, List

from tests.globals.utils import NUMBER_OF_DOCUMENTS


@pytest.fixture(scope="session")
def numpy_documents(numpy_document: Dict) -> List:
    return [numpy_document for _ in range(NUMBER_OF_DOCUMENTS)]
