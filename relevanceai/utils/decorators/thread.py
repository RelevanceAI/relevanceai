import threading


def thread(f):
    """
    Use as such:

    Example
    ----------

    .. code-block::

        @fire_and_forget
        def send_analytics():
            ...
    """

    def wrapped():
        threading.Thread(target=f).start()

    return wrapped
