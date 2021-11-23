# How to build docs

On Unix/Linux systems, run:

```
make build_docs
```

For questions, contact Jacky Wong.

## Installation 

Run:

```
pip install -e .. && pip install sphinx sphinx-rtd-theme && python -m pip install sphinx-autoapi
```

## Re-creating docs

In order to re-create the documentation, 

run `make_build` and, look at every RST created by automodule and then remove unnecessary subheadings.
