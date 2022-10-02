import argparse
from relevanceai.utils import decode_workflow_token


def read_token_from_script():
    """
    Reads in a token from script and returns a config as a
    dictionary object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("token", help="The token used for the workflow config.")
    args = parser.parse_args()
    token = args.token
    config = decode_workflow_token(token)
    return config
