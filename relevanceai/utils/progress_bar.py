"""Get a good progress bar
"""
# from contextlib import nullcontext
from contextlib import AbstractContextManager


class ProgressBar:
    def __call__(self, iterable):
        return self.get_bar()(iterable)

    @staticmethod
    def is_in_ipython():
        """
        Determines if current code is executed within an ipython session.
        """
        is_in_ipython = False
        # Check if the runtime is within an interactive environment, i.e., ipython.
        try:
            from IPython import get_ipython  # pylint: disable=import-error

            if get_ipython():
                is_in_ipython = True
        except ImportError:
            pass  # If dependencies are not available, then not interactive for sure.
        return is_in_ipython

    def is_in_notebook(self) -> bool:
        """
        Determines if current code is executed from an ipython notebook.
        """
        is_in_notebook = False
        if self.is_in_ipython():
            # The import and usage must be valid under the execution path.
            from IPython import get_ipython

            if "IPKernelApp" in get_ipython().config:
                is_in_notebook = True
        return is_in_notebook

    def get_bar(self):
        from tqdm.auto import tqdm

        return tqdm


class NullProgressBar(AbstractContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, iterable: int = None):
        self.iterable = iterable

    def __enter__(self):
        return self.iterable

    def __exit__(self, *excinfo):
        pass

    def __iter__(self):
        if self.iterable is None:
            self.iterable = range(0)
        for i in self.iterable:
            yield i


def progress_bar(iterable, show_progress_bar: bool = False):

    try:
        if show_progress_bar:
            return ProgressBar()(iterable)
    except Exception as e:
        pass
    return NullProgressBar(iterable)
