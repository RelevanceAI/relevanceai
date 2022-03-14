from functools import wraps


def list_to_tuple(function):
    @wraps(function)
    def wrapper(*args):
        args = [tuple(x) if type(x) == list else x for x in args]
        result = function(*args)
        result = tuple(result) if type(result) == list else result
        return result

    return wrapper
