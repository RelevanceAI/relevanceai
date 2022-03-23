import warnings

from functools import wraps

from relevanceai.constants.messages import Messages


def beta(f):
    f.__doc__ = Messages.BETA_DOCSTRING.format(f.__doc__)

    @wraps(f)
    def wrapper(*args, **kwds):
        return f(*args, **kwds)

    return wrapper


def added(version):
    def _version(f):
        f.__doc__ = Messages.ADDED_DOCSTRING.format(version)

        @wraps(f)
        def wrapper(*args, **kwds):
            return f(*args, **kwds)

        return wrapper

    return _version


def deprecated(version: str, message: str = ""):
    def _version(f):
        f.__doc__ = Messages.DEPRECEATED_DOCSTRING.format(version)

        @wraps(f)
        def wrapper(*args, **kwds):
            warnings.warn(
                Messages.DEPRECEATED.format(version, message), DeprecationWarning
            )
            return f(*args, **kwds)

        return wrapper

    return _version
