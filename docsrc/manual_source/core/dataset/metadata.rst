Metadata
===========

You can also easily access metadata using the following

.. code-block::

    import time
    ds = client.Dataset("_mock_dataset_")
    ds.metadata['value'] = 3
    ds.metadata['strong_values'] = 10
    time.sleep(1)
    ds.metadata
    # returns
    # {"value": 3, "strong_values": 10}
