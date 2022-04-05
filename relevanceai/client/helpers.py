import getpass

from dataclasses import dataclass
from typing import List

from relevanceai.constants.messages import Messages
from relevanceai.constants import (
    SIGNUP_URL,
    AUSTRALIA_URL,
    WIDER_URL,
    OLD_AUSTRALIA_EAST,
    DEV_URL,
)
from relevanceai.constants.errors import (
    APIKeyNotFoundError,
    FireBaseUIDNotFoundError,
    ProjectNotFoundError,
    RegionNotFoundError,
    TokenNotFoundError,
)


def region_to_url(region: str) -> str:
    """A function to convert a users region to base url for API calls

    Args:
        region (str): user region, element of ["us-east-1", "ap-southeast-2", "old-australia-east"]

    Returns:
        url: the appropriate base url for API calls
    """
    if "dev" in region:
        actual_region = region.replace("dev-", "")
        url = DEV_URL.format(actual_region)

    elif region == OLD_AUSTRALIA_EAST:
        url = AUSTRALIA_URL

    else:
        url = WIDER_URL.format(region)

    return url


@dataclass
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

    def split_token(self) -> List[str]:
        """
        Splits the token in its four main components given in order by above

        Returns:
            List[str]: a list of strings containing all user identification for interaction with dashboard and api calls
        """
        return self.token.split(":")

    def dict(self) -> dict:
        return {
            "project": self.project,
            "api_key": self.api_key,
            "region": self.region,
            "firebase_uid": self.firebase_uid,
            "token": self.token,
        }


def process_token(token: str):
    """Given a user token, checks to see if all necessary credentials are present in token.

    Args:
        token (str): a ":" delimited string of identifying information

    Raises:
        TokenNotFoundError: error if idenitifier is not found
        ProjectNotFoundError: error if idenitifier is not found
        APIKeyNotFoundError: error if idenitifier is not found
        RegionNotFoundError: error if idenitifier is not found
        FireBaseUIDNotFoundError: error if idenitifier is not found

    Returns:
        Credentials: a dataclass of all user identification
    """
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


def auth() -> str:
    """_summary_

    Returns:
        token: a ":" delimited string of identifying information
    """
    print(Messages.TOKEN_MESSAGE.format(SIGNUP_URL))
    token = getpass.getpass(f"Activation Token: ")
    return token
