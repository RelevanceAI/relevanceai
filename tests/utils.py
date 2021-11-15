import random
import string

def generate_random_string(N=5):
    """Generate a random string of letters and numbers
    """
    return ''.join(random.choice(
        string.ascii_uppercase + string.digits) \
            for _ in range(N))
