# How to build docs


For questions, contact Jacky Wong.

## Installation 

Run:

```
pip install -e .. && pip install sphinx sphinx-rtd-theme && python -m pip install sphinx-autoapi & pip install sphinx-autodoc-typehints
```

On Unix/Linux systems, then run:

```
make build_docs
```

## Re-creating docs

In order to re-create the documentation, 

run `make_build` and, look at every RST created by automodule and then remove unnecessary subheadings.

You wll also want to add this fix to prevent sphinx erroring:  

https://github.com/sphinx-doc/sphinx/issues/1453
