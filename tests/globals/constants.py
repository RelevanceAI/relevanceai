import random
import string

from typing import List

VECTOR_LENGTH = 8
NUMBER_OF_DOCUMENTS = 100


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
