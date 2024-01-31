import os
import json
import getpass


class Config:
    def __init__(self):
        self._auth = None
        self.headers = {}

    @property
    def auth(self):
        if not self._auth:
            login()
        return self._auth

    def set_auth(self, auth):
        self._auth = auth


config = Config()


class Auth:
    def __init__(self, api_key: str, region: str, project: str):
        self.api_key = api_key
        self.region = region
        self.project = project
        self.headers = {"Authorization": f"{project}:{api_key}"}
        self.url = f"https://api-{self.region}.stack.tryrelevance.com"


def login(
    api_key: str = None, region: str = None, project: str = None, store: bool = True
):
    cred_path = os.path.expanduser("~/relevanceai.json")
    cred_json = {}
    try:
        if os.path.exists(cred_path):
            with open(cred_path, "r") as f:
                cred_json = json.load(f)
    except:
        print("Error reading credentials file, please login again")

    if cred_json:
        if api_key and region and project:
            cred_json = {"api_key": api_key, "region": region, "project": project}
    else:
        print(
            "You can create and find your API key in your browser here: https://app.relevanceai.com/login/sdk"
        )
        if not api_key:
            api_key = os.getenv("RELEVANCE_API_KEY") or getpass.getpass(
                "Paste the API key from your profile and hit enter: "
            )

        if not region:
            region = os.getenv("RELEVANCE_REGION") or input(
                "Paste the Region from your profile and hit enter: "
            )

        if not project:
            project = os.getenv("RELEVANCE_PROJECT") or input(
                "Paste the Project from your profile and hit enter: "
            )

        cred_json = {"api_key": api_key, "region": region, "project": project}
    config.set_auth(Auth(**cred_json))
    # TBC: make it more secure with netsrc
    if store:
        with open(cred_path, "w") as f:
            json.dump(cred_json, f)
        os.chmod(cred_path, 0o600)
    print("Successfully logged in, welcome!")
