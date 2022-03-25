import warnings
from functools import wraps


def beta(f):
    old_doc = f.__doc__
    f.__doc__ = (
        old_doc
        + """

.. warning::
    This function is currently in beta and is liable to change in the future. We
    recommend not using this in production systems.

    """
    )

    @wraps(f)
    def wrapper(*args, **kwds):
        warnings.warn("This function currently in beta and may change in the future.")
        return f(*args, **kwds)

    return wrapper


def introduced_in_version(version_number):
    def _version(f):
        old_doc = f.__doc__
        new_doc = (
            old_doc
            + f"""

.. note::
    This function was introduced in **{version_number}**.

        """
        )
        f.__doc__ = new_doc

        @wraps(f)
        def wrapper(*args, **kwds):
            return f(*args, **kwds)

        return wrapper

    return _version


def deprecated(version_number: str, message: str = ""):
    def _version(f):
        old_doc = f.__doc__
        new_doc = (
            old_doc
            + f"""

.. note::
    This function has been deprecated as of {version_number}

        """
        )
        f.__doc__ = new_doc

        @wraps(f)
        def wrapper(*args, **kwds):
            DEPRECATION_MESSAGE = (
                f"Deprecated. Revert to versions before {version_number} for function. "
                + message
            )
            warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning)
            return f(*args, **kwds)

        return wrapper

    return _version
