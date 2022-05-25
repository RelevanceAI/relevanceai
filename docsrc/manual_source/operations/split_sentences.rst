Split Sentences
=================

You can split sentences using a simple function

.. code-block::

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("sample")
    ds.split_sentences(
        text_fields=["sample"]
    )

For more fine-grained control, you can use the natural operator:

.. code-block::

    from relevanceai.operations_new.processing.text.sentence_splitting.ops import (
        SentenceSplitterOps,
    )

    ops = SentenceSplitterOps(language=language)
    for c in self.chunk_dataset(select_fields=text_fields):
        for text_field in text_fields:
            c = ops.run(
                text_field=text_field,
                documents=c,
                inplace=True,
                output_field=output_field,
            )
        self.upsert_documents(c)

If you want to split sentences infinitely, you can simply use this:

.. code-block::

    classs NewSentenceSplitter(SentenceSplitterOps):
        def split_text(self, text):
            # the return MUST be a list of texts
            return ["text_section_1", "text_section_2"]
