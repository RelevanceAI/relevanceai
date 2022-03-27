import getpass

from relevanceai.constants.messages import Messages
from relevanceai.constants import (
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

    if len(split_token) == 4:
        region = split_token[2]
        firebase_uid = split_token[3]
        data = dict(
            project=project,
            api_key=api_key,
            region=region,
            firebase_uid=firebase_uid,
        )

    elif len(split_token) == 3:
        region = split_token[2]
        data = dict(
            project=project,
            api_key=api_key,
            region=region,
        )

    else:
        data = dict(
            project=project,
            api_key=api_key,
        )

    return data


def auth():
    print(Messages.TOKEN_MESSAGE.format(SIGNUP_URL))
    token = getpass.getpass(f"Activation Token: ")
    return token
