from http.client import NOT_IMPLEMENTED
import os
import random
import string

from typing import List

TEST_TOKEN = os.getenv("TEST_TOKEN")
if TEST_TOKEN is None:
    PROJECT = os.getenv("TEST_PROJECT")
    API_KEY = os.getenv("TEST_API_KEY")

    if PROJECT is None and API_KEY is None:
        raise ValueError("Set Env Var TEST_TOKEN")
    else:
        TEST_TOKEN = f"{PROJECT}:{API_KEY}:us-east-1:{None}"

TEST_TOKEN = TEST_TOKEN.split(":")
TEST_FIREBASE_UID = "relevanceai-sdk-test-user"
TEST_TOKEN[-1] = TEST_FIREBASE_UID
TEST_TOKEN = ":".join(TEST_TOKEN)


VECTOR_LENGTH = 8
NUMBER_OF_DOCUMENTS = int(os.getenv("TEST_NUMBER_OF_DOCUMENTS", 20))


def generate_random_string(string_length: int = 5) -> str:
    """Generate a random string of letters and numbers"""
    return "".join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(string_length)
    )


def generate_random_vector(vector_length: int = VECTOR_LENGTH) -> List[float]:
    """Generate a random list of floats"""
    return [random.random() for _ in range(vector_length)]


def generate_random_label(label_value: int = 5) -> str:
    return f"label_{random.randint(0, label_value)}"


def generate_random_integer(min: int = 0, max: int = 100) -> int:
    return random.randint(min, max)


def generate_dataset_id():
    return SAMPLE_DATASET_DATASET_PREFIX + generate_random_string().lower()


RANDOM_PANDAS_DATASET_SUFFIX = generate_random_string().lower()
SAMPLE_DATASET_DATASET_PREFIX = "_sample_test_dataset_"
LABEL_DATSET_ID = SAMPLE_DATASET_DATASET_PREFIX + generate_random_string().lower()

NOT_IMPLEMENTED = "Test Not Implemented Yet"
