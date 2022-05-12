Document Utilities
========

With Relevance AI, the preferred way to locally interact with
a document/list of documents is with DocUtils

If you need to get a field

.. code-block::

    from relevanceai.utils import DocUtils

    document = {"field1": 1, "field2": 2}

    value = DocUtils.get_field("field1", document)

If you need to set a field, you can do so either inplace or not. Default is inplace

.. code-block::

    document = {"field1": 1, "field2": 2}

    DocUtils.set_field("field1", document, 3) # edits original document

    new_document = DocUtils.set_field("field1", document, 3, inplace=False) # returns an editted copy
