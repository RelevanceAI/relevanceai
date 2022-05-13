"""A cache mixin
"""


def _is_cache_function(func):
    """Determines if a function is cached or not"""
    return hasattr(func, "cache_info")


def rdir_cache_functions(base_function, function_store: set = None):
    """Recursively gets all the cached functions"""
    if function_store is None:
        function_store = set()
    all_attrs = dir(base_function)
    for attr in all_attrs:
        func = getattr(base_function, attr)
        # Check if it is a class but not that it is a global variable
        # class is denoted by capitalised first letter
        # global variable is denoted by all upper case
        res = _is_cache_function(func)
        if attr[0].isupper() and not attr.upper() == attr:
            rdir_cache_functions(func, function_store=function_store)
        # check for internal classes
        elif attr[0] == "_" and attr[1].isupper() and not attr.upper() == attr:
            rdir_cache_functions(func, function_store=function_store)
        elif res:
            function_store.add(func)
    return function_store


class CacheMixin:
    def _get_all_cache_functions(self):
        import relevanceai

        cache_functions = set()
        for directory in [
            relevanceai.client.client,
            relevanceai.client.client.Dataset,
            relevanceai.dataset.series.Series,
        ]:
            cache_functions.update(rdir_cache_functions(directory))
        return cache_functions

    def clear_cache(self, *args, **kw):
        """
        Clears cache.

        .. code-block::

            from relevanceai import Client
            client = Client()
            client.cache_info()

        """
        cache_functions = self._get_all_cache_functions()
        for func in cache_functions:
            if hasattr(func, "clear_cache"):
                func.clear_cache(*args, **kw)

    def cache_info(self):
        """
        Prints statements explaining how much is stored in each cache.

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            client.cache_info()

        """
        cache_functions = self._get_all_cache_functions()
        for func in cache_functions:
            if hasattr(func, "cache_info"):
                print(func)
                info = func.cache_info()
                print(info)
