# RelevanceAI

For guides, tutorials on how to use this package, visit https://docs.relevance.ai/docs.

If you are looking for an SDK reference, you can find that [here](https://youthful-leakey-ab1977.netlify.app/index.html).

Built mainly for users looking to experiment with vectors/embeddings.

## Installation 

The easiest way is to install this package is to run `pip install --upgrade relevanceai`.

## How to use the RelevanceAI client

For the relevanceai client, we want to ensure the SDK mirrors the API client.

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

Otherwise, you can also setup your dev env using the [`Makefile`](./Makefile)

Setup your virtualenv, install requirements and package

```python
❯ make install
```

For all available targets

```python
❯ make help
Available rules:
clean               Delete all compiled Python files 
install             Install dependencies 
lint                Lint using flake8 
test                Test dependencies 
update              Update dependencies 
```

To add new targets, add new command and intended script, add descriptive comment above the command block as description to the rule.

```make
#################################################################################
# COMMANDS                                                                      #
#################################################################################
...
## Test dependencies
test
	pytest $(TEST_PATH) --cov=relevanceai -vv -rs -x
...
```
Then run testing using:

You can limit your testing on a single file/folder by specifying a test path to folder or file.


```zsh

❯ make test TEST_PATH=tests/integration

```

