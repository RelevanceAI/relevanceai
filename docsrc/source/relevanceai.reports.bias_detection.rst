Bias Detection
=================

Users can detect bias in a model by seeing which concepts certain vectors are closer to.
This is a particularly useful tool when users are looking at semantic vectors and
would like to check if certain words are leaning particularly towards any 
specific category.

An example of analysing gender bias inside Google's Universal Sentence Encoder 
can be found below.

.. code-block::

    # Set up the encoder
    !pip install -q vectorhub[encoders-text-tfhub]
    from vectorhub.encoders.text.tfhub import USE2Vec
    enc = USE2Vec()

    from relevanceai.bias_detection import bias_indicator

    bias_indicator(
        ["boy", "girl"], # the categories of bias
        ["basketball", "draft", "skirt", "dress", "grave digger"], # words to care about
        enc.encode
    )

