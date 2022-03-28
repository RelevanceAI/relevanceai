import getpass

from dataclasses import dataclass

from relevanceai.constants.messages import Messages
from relevanceai.constants import (
    SIGNUP_URL,
    AUSTRALIA_URL,
    WIDER_URL,
    OLD_AUSTRALIA_EAST,
)
from relevanceai.constants.errors import (
    APIKeyNotFoundError,
    FireBaseUIDNotFoundError,
    ProjectNotFoundError,
    RegionNotFoundError,
    TokenNotFoundError,
)


def region_to_url(region: str):
    if region == OLD_AUSTRALIA_EAST:
        url = AUSTRALIA_URL
    else:
        url = WIDER_URL.format(region)
    return url


@dataclass(frozen=True)
class Credentials:
    """
    A convenience store of relevant credentials.
    """

    __slots__ = (
        "token",
        "project",
        "api_key",
        "region",
        "firebase_uid",
    )
    token: str
    project: str
    api_key: str
    region: str
    firebase_uid: str

    def split_token(self):
        return self.token.split(":")


def process_token(token: str):
    if not token:
        raise TokenNotFoundError

    # A token takes the format project:api_key:region:firebase_uid
    project, api_key, *other_credentials = token.split(":")

    if not project:
        raise ProjectNotFoundError

    if not api_key:
        raise APIKeyNotFoundError

    if len(other_credentials) == 0:
        # Assume that region is missing as that is the next credential
        raise RegionNotFoundError
    elif len(other_credentials) == 1:
        # Assume region exists but firebase_uid, the next credential, does not
        raise FireBaseUIDNotFoundError
    else:
        # Two or more other credentials is correct
        region, firebase_uid, *additional_credentials = other_credentials

    return Credentials(token, project, api_key, region, firebase_uid)


def auth():
    print(Messages.TOKEN_MESSAGE.format(SIGNUP_URL))
    token = getpass.getpass(f"Activation Token: ")
    return token
