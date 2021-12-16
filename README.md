# RelevanceAI

For guides, tutorials on how to use this package, visit https://docs.relevance.ai/docs.

If you are looking for an SDK reference, you can find that [here](https://relevanceai.github.io/RelevanceAI/docs/html/index.html).

Built mainly for data scientists/engineers looking to experiment with vectors/embeddings.

## Installation 

The easiest way is to install this package is to run `pip install --upgrade relevanceai`.

You can also install on conda: `conda install -c relevance relevanceai`.

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

## Config

The config contains the adjustable global settings for the SDK. For a description of all the settings, see [here](https://relevanceai.github.io/RelevanceAI/docs/html/index.html).  

To view setting options, run the following:

```python
client.config.options
```

The syntax for selecting an option is *section.key*. For example, to disable logging, run the following to modify *logging.enable_logging*:

```python
client.config.set_option('logging.enable_logging', False)
```

To restore all options to their default, run the following:

```python
client.config.reset_to_default()
```
