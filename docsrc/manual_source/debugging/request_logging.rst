Request Logging
==================

Toggle
-------------------

To turn on/off request logging, set the environment variable DEBUG_REQUESTS to either TRUE or FALES (captilised string)

.. code-block::

    import os
    os.environ["DEBUG_REQUESTS"] = "TRUE" # will log all requests and data to "request.log"

    from relevanceai import Client
    client = Client()
    df = client.Dataset("sample_dataset_id")

    csv_filename = "temp.csv"
    df.insert_csv(csv_filename)
