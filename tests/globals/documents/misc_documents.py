import pytest

from tests.globals.document import DataclassDocument


@pytest.fixture(scope="session", autouse=True)
def test_dataclass_documents():

    return [DataclassDocument()] * 20
