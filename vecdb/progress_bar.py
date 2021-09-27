"""Get a good progress bar
"""
# from contextlib import nullcontext
from contextlib import AbstractContextManager

class ProgressBar:
    def __call__(self, *args, **kwargs):
        self.logger.info("WHAT BAR")
        return self.get_bar()(*args, **kwargs)
    
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
            if 'IPKernelApp' in get_ipython().config:
                is_in_notebook = True
        return is_in_notebook
    
    def get_bar(self):
        if self.is_in_notebook():
            from tqdm.notebook import tqdm as notebook_bar
            return notebook_bar
        from tqdm import tqdm as normal_bar
        return normal_bar

class NullProgressBar(AbstractContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, enter_result: int=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass

    def __iter__(self):
        if self.enter_result is None:
            self.enter_result = 999999999999999
        for i in range(self.enter_result):
            yield i

def progress_bar(*args, show_progress_bar: bool=False, **kwargs):
    try:
        if show_progress_bar:
            return ProgressBar()(*args, **kwargs)
    except:
        return NullProgressBar()
    return NullProgressBar()
