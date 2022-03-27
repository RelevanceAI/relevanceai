from functools import wraps


def list_to_tuple(function):
    @wraps(function)
    def wrapper(*args, **kw):
        args = [tuple(x) if type(x) == list else x for x in args]
        new_kw = {}
        for k, v in kw.items():
            if type(v) == list:
                new_kw[k] = tuple(v)
            else:
                new_kw[k] = v
        result = function(*args, **new_kw)
        result = tuple(result) if type(result) == list else result
        return result

    return wrapper
