Transformers Pipelines
========================

Transformers pipelines allow everyone to easily add functionality.

It is as simple as just applying the pipeline once it is instantiated.

This allows users to:

- Store metadata about the pipeline automatically
- Automatically apply a number of HuggingFace pipelines with ease

Running it on a dataset:

.. code-block::

    from transformers import pipeline
    pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
    ds.apply_transformers_pipeline(
        text_fields, pipeline
    )
