import pytest

from relevanceai import Client

from test_integrations.constants import (
    REGION,
    SAMPLE_DATASET_DATASET_PREFIX,
    TEST_API_KEY,
    TEST_PROJECT,
    TEST_US_API_KEY,
    TEST_US_PROJECT,
)
from test_integrations.utils import generate_dataset_id


@pytest.fixture(scope="session")
def test_project():
    if REGION == "us-east-1":
        return TEST_US_PROJECT
    return TEST_PROJECT


@pytest.fixture(scope="session")
def test_api_key():
    if REGION == "us-east-1":
        return TEST_US_API_KEY
    return TEST_API_KEY


@pytest.fixture(scope="session")
def test_firebase_uid():
    return "relevanceai-sdk-test-user"


@pytest.fixture(scope="session")
def test_client(test_project: str, test_api_key: str, test_firebase_uid: str):
    if REGION is None:
        client = Client(
            project=test_project, api_key=test_api_key, firebase_uid=test_firebase_uid
        )
    else:
        client = Client(
            project=test_project,
            api_key=test_api_key,
            firebase_uid=test_firebase_uid,
            region=REGION,
        )

    # For some reason not resetting to default
    # correct_client_config(client)
    client.config["mixpanel.is_tracking_enabled"] = False
    client.disable_analytics_tracking()
    yield client

    # While unnecessary since test_dataset is deleted upon return, put in
    # place for future use.
    for dataset in client.datasets.list()["datasets"]:
        if dataset.startswith(SAMPLE_DATASET_DATASET_PREFIX):
            client.delete_dataset(dataset)


# scope set to function so that a new ID is generated for each test
@pytest.fixture(scope="function")
def test_dataset_id():
    return generate_dataset_id()


@pytest.fixture(scope="session")
def test_documents():
    return [
        {
            "_id": "1",
            "data": "xkcd_existential",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/b/b5/Xkcd_philosophy.png",
        },
        {
            "_id": "2",
            "data": "comic_1",
            "image_url": "https://lumiere-a.akamaihd.net/v1/images/maractsminf001_cov_2a89b17b.jpeg?region=0%2C0%2C1844%2C2800",
        },
        {
            "_id": "3",
            "data": "comic_2",
            "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTo-j3JHpQMonPr4WW4iu8hizI4mzYsD_xi9w&usqp=CAU",
        },
        {
            "_id": "4",
            "data": "comic_3",
            "image_url": "https://lumiere-a.akamaihd.net/v1/images/maractsminf011_cov_d4e503b7.jpeg?region=0%2C0%2C1844%2C2800",
        },
        {
            "_id": "6",
            "data": "pig",
            "image_url": "https://static.scientificamerican.com/sciam/cache/file/51126F79-1EA3-40F4-99D832BADE5D0156.jpg",
        },
        {
            "_id": "8",
            "data": "gorilla",
            "image_url": "https://static.scientificamerican.com/sciam/cache/file/8DCE99C5-34B1-44FA-AF07CD37C58F18B2.jpg",
        },
        {
            "_id": "10",
            "data": "monkey",
            "image_url": "https://static.scientificamerican.com/sciam/cache/file/4A7A86B9-3BC1-43D1-9097B71758E1C11A_source.jpg?w=590&h=800&F2750780-CA0A-4AF0-BC4484CBF331C802",
        },
        {
            "_id": "11",
            "data": "eagle",
            "image_url": "https://static.scientificamerican.com/sciam/cache/file/2BE2A480-FE3F-4E6C-AAB3E8BED95CEC56_source.jpg",
        },
        {
            "_id": "5",
            "data": "bird",
            "image_url": "https://static.scientificamerican.com/sciam/cache/file/7A715AD8-449D-4B5A-ABA2C5D92D9B5A21_source.png",
        },
    ]
