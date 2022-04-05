"""Test utilities
"""


def is_subname_in_schema(name, schema):
    schema_values = list(schema)
    return any([name in x for x in schema_values])


def correct_client_config(client):
    client.config.reset()
    if client.region != "us-east-1":
        raise ValueError("default value aint RIGHT")
