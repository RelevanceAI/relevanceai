# RelevanceAI

For guides, tutorials on how to use this package, visit https://docs.relevance.ai/docs.

If you are looking for an SDK reference, you can find that [here](https://youthful-leakey-ab1977.netlify.app/index.html).

Built mainly for users looking to experiment with vectors/embeddings.

## Installation 

The easiest way is to install this package is to run `pip install --upgrade relevanceai`.

## How to use the RelevanceAI client

For example:

```python
## To instantiate the client 
from relevanceai import Client
client = Client()
```

## Development

### Getting Started

To get started with development, ensure you have `pytest` and `mypy` installed. These will help ensure typechecking and testing.

```python
python -m pip install pytest mypy
```


Then run testing using:

Make sure to set your test credentials!

```
export TEST_PROJECT = xxx 
export TEST_API_KEY = xxx 
```

```python
python -m pytest
mypy relevanceai
```
