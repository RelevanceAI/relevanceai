Label
==========

Users can label their datasets with `label_documents`.

Labelling with `label_documents` can be done using:

.. code-block::

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("sample")
    label = [
        {
            "label": "sample_label",
            "label_vector_": [1, 2, 1, 2, 1]
        },
        {
            "label": "sample_label_2",
            "label_vector_": [1, 1, 1, 1, 1]
        }
    ]
    ds.label(
        vector_fields=["sample_1_vector_"],
        label_documents=label_documents,
        expanded=True # If you just want a list of tags and not extra information, set `expanded=False`
    )

.. automethod:: relevanceai.operations_new.ops.Operations.label
