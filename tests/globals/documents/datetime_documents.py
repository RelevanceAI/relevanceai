import pytest

from typing import Dict, List

from tests.globals.utils import NUMBER_OF_DOCUMENTS


@pytest.fixture(scope="session")
def datetime_documents(datetime_document: Dict) -> List:
    return [datetime_document for _ in range(NUMBER_OF_DOCUMENTS)]
