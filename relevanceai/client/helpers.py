from relevanceai.constants.messages import Messages
from relevanceai.constants.constants import (
    SIGNUP_URL,
    AUSTRALIA_URL,
    WIDER_URL,
    OLD_AUSTRALIA_EAST,
)


def region_to_url(region: str):
    if region == OLD_AUSTRALIA_EAST:
        url = AUSTRALIA_URL
    else:
        url = WIDER_URL.format(region)
    return url


def process_token(token: str):
    split_token = token.split(":")

    project = split_token[0]
    api_key = split_token[1]
    region = split_token[2]
    firebase_uid = split_token[3]

    data = dict(
        project=project,
        api_key=api_key,
        region=region,
        firebase_uid=firebase_uid,
    )
    return data


def token_to_auth(token: str):
    if token:
        return process_token(token)
    else:
        print(Messages.TOKEN_MESSAGE.format(SIGNUP_URL))
        return process_token(token)