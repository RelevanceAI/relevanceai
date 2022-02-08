import pytest

from tests.globals.document import _sample_datetime_document

from tests.globals.utils import NUMBER_OF_DOCS


@pytest.fixture(scope="session", autouse=True)
def sample_datetime_documents():
    return [_sample_datetime_document() for _ in range(NUMBER_OF_DOCS)]
