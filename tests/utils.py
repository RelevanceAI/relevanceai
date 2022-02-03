import random
import string
from typing import List


def generate_random_string(N: int = 5) -> str:
    """Generate a random string of letters and numbers"""
    return "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(N)
    )


def generate_random_vector(N: int = 512) -> List[float]:
    """Generate a random list of floats"""
    return [random.random() for _ in range(N)]


def generate_random_label(N: int = 5) -> str:
    return f"label_{random.randint(0, N)}"


def generate_random_integer(N: int = 100) -> int:
    return random.randint(0, N)
