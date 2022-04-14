from functools import wraps

from relevanceai.utils.logger import FileLogger


def log(fn: str = "logs.txt", verbose: bool = False, log_to_file=True):
    def _log(f):

        with FileLogger(fn=fn, verbose=verbose, log_to_file=log_to_file):

            @wraps(f)
            def wrapper(*args, **kwds):
                return f(*args, **kwds)

        return wrapper

    return _log
