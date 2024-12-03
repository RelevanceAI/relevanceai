from __future__ import annotations

from .._client import RelevanceAI
from .._resource import SyncAPIResource

from ..resources.agent import Agent
from typing import List


class Agents(SyncAPIResource):

    _client: RelevanceAI

    def list_agents(self) -> List[Agent]:
        path = "agents/list"
        response = self._post(path)
        agents = [
            Agent(client=self._client, **item)
            for item in response.json().get("results", [])
        ]
        return sorted(
            agents, key=lambda x: (x.metadata.name is None, x.metadata.name or "")
        )

    def retrieve_agent(
        self,
        agent_id: str,
    ) -> Agent:
        path = f"agents/{agent_id}/get"
        response = self._get(path)
        return Agent(client=self._client, **response.json()["agent"])
