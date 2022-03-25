import random
import string

from test_integrations.constants import SAMPLE_DATASET_DATASET_PREFIX


def _generate_random_string(string_length: int = 5) -> str:
    """Generate a random string of letters and numbers"""
    return "".join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(string_length)
    )


def generate_dataset_id():
    return SAMPLE_DATASET_DATASET_PREFIX + _generate_random_string().lower()
