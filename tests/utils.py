"""Test utilities
"""


def is_subname_in_schema(name, schema):
    return any([name in x for x in schema])
