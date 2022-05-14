Cache
========

With Relevance AI, caching automatically happens in a few
very common situations: 

- When you are retrieving all documents (caches documents)
- When you are instantiating large models (caches models)

However, if you need to clear cache, you can do so using:

.. code-block::

    from relevanceai import Client
    client = Client()
    client.clear_cache()

You can also get cache info using:

.. code-block::

    client.cache_info()

Under the hood, it recursively gets all the functions that
are cached and then returns the relevant cache information.

Caching Algorithm 
--------------------

The caching algorithm is a slightly modified version of
Python's default LRU but with an updated hashing algorithm 
that stringifies lists and dictionaries as part of the key. 
