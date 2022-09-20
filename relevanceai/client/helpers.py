import getpass

from dataclasses import dataclass
from typing import List

from relevanceai.constants.messages import Messages
from relevanceai.constants import SIGNUP_URL


def region_to_url(region: str) -> str:
    """A function to convert a users region to base url for API calls

    Args:
        region (str): user region, element of ["us-east-1", "ap-southeast-2", "old-australia-east"]

    Returns:
        url: the appropriate base url for API calls
    """
    return f"https://api-{region}.stack.relevance.ai/latest/"


@dataclass
class Credentials:
    """
    A convenience store of relevant credentials.
    """

    auth_token: str
    project: str
    api_key: str
    url_or_region: str
    token: str
    refresh_token: str = None

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
            "url_or_region": self.url_or_region,
            "auth_token": self.auth_token,
            "refresh_token": self.refresh_token,
            "session_token": self.token,
        }


def process_token(token):
    """
    Tokens come in the form of:
    project:api_key:url_or_region
    If there is a fourth one - that's for mixpanel to identify the firebase user
    We are going to replace the api key call with the session_token
    """
    split_token = token.split(":")

    data = {
        "project": split_token[0],
        "seession_token": split_token[1],
        "url_or_region": split_token[2]
    }

    if len(split_token) > 3:
        data["session_token"] = split_token[3]
    else:
        data["session_token"] = None

    if len(split_token) > 4:
        data["refresh_token"] = split_token[4]
    else:
        data["refresh_token"] = None

    return Credentials(
        auth_token=token,
        project=data["project"],
        api_key=data["api_key"],
        url_or_region=data["url_or_region"],
        token=data["session_token"],
        refresh_token=data["refresh_token"],
    )


def auth() -> str:
    """_summary_

    Returns:
        token: a ":" delimited string of identifying information
    """
    print(Messages.TOKEN_MESSAGE.format(SIGNUP_URL))
    token = getpass.getpass(f"Activation Token: ")
    return token
