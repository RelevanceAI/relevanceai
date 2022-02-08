import pytest

from typing import Dict, List

from tests.globals.utils import NUMBER_OF_DOCUMENTS


@pytest.fixture(scope="session")
def pandas_documents(pandas_document: Dict) -> List:
    return [pandas_document for _ in range(NUMBER_OF_DOCUMENTS)]
