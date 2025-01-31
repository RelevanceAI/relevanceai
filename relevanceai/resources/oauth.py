from __future__ import annotations

from .._client import RelevanceAI, AsyncRelevanceAI
from .._resource import SyncAPIResource, AsyncAPIResource

from ..resources.agent import Agent, AsyncAgent
from typing import List, Optional
from ..types.oauth import *
import webbrowser


class OAuthManager(SyncAPIResource):

    _client: RelevanceAI

    def add_google_integration(self, auto_open: bool = False) -> str:
        path = "auth/oauth/get_url"
        body = {"provider": "google", "types": ["email-read-write"], "redirect_url": ""}
        response = self._post(path, body=body)
        response = OAuth(**response.json())
        if auto_open:
            webbrowser.open(response.auth_url)
        return response.auth_url

    def list_active_integrations(self):
        path = "auth/oauth/accounts/list"
        response = self._post(path)
        # return response.json()
        return ActiveIntegrations(**response.json())
