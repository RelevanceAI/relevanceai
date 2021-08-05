"""Get a good progress bar
"""

class ProgressBar:
    def __call__(self, *args, **kwargs):
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

def progress_bar(*args, **kwargs):
    try:
        return ProgressBar()(*args, **kwargs)
    except:
        return
