Operators
------------

Operators run on a dataset and interact with Datasets in a
streamlined fashion.

Operators have the following:

- All parameters of an operator are defined in `__init__.py`.
- All operators are have a `transform` method that shows What
happens when you accept documents and return documents
- Operators are also written in `ops.py` files that inherit
from an OperationsAPIBase.

.. toctree::
    :maxdepth: 4

    Cluster <cluster/index>
    dim_reduction
    vectorize
    sentiment
    cluster_viz
    search
    label
    question_answer
    split_sentences
    Finetuning <finetuning/index>
