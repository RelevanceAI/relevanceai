from __future__ import annotations

from .._client import RelevanceAI, AsyncRelevanceAI
from .._resource import SyncAPIResource, AsyncAPIResource

from ..resources.agent import Agent, AsyncAgent
from typing import List, Optional


class AgentsManager(SyncAPIResource):

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

    def create_agent(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[int] = None,
    ) -> Agent:
        existing_agents = self.list_agents()
        for agent in existing_agents:
            if agent.metadata.name == name:
                raise ValueError(f"Agent with name '{name}' already exists!")
        path = f"/agents/upsert"
        body = {
            k: v
            for k, v in {
                "name": name,
                "system_prompt": system_prompt,
                "model": model,
                "temperature": temperature,
            }.items()
            if v is not None
        }
        response = self._post(path, body=body)
        agent = self.retrieve_agent(response.json()["agent_id"])
        return agent

    def upsert_agent(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[int] = None,
    ) -> Agent:
        path = f"/agents/upsert"
        body = {
            k: v
            for k, v in {
                "agent_id": agent_id,
                "name": name,
                "system_prompt": system_prompt,
                "model": model,
                "temperature": temperature,
            }.items()
            if v is not None
        }
        response = self._post(path, body=body)
        agent = self.retrieve_agent(response.json()["agent_id"])
        return agent

    def delete_agent(
        self,
        agent_id: str,
    ) -> bool:
        path = f"agents/{agent_id}/delete"
        response = self._post(path)
        return response.status_code == 200


class AsyncAgentsManager(AsyncAPIResource):
    _client: AsyncRelevanceAI

    async def list_agents(self) -> List[AsyncAgent]:
        path = "agents/list"
        response = await self._post(path)
        agents = [
            AsyncAgent(client=self._client, **item)
            for item in (response.json()).get("results", [])
        ]
        return sorted(
            agents, key=lambda x: (x.metadata.name is None, x.metadata.name or "")
        )

    async def retrieve_agent(
        self,
        agent_id: str,
    ) -> AsyncAgent:
        path = f"agents/{agent_id}/get"
        response = await self._get(path)
        return AsyncAgent(client=self._client, **response.json()["agent"])

    async def upsert_agent(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[int] = None,
    ) -> AsyncAgent:
        path = f"/agents/upsert"
        body = {
            k: v
            for k, v in {
                "agent_id": agent_id,
                "name": name,
                "system_prompt": system_prompt,
                "model": model,
                "temperature": temperature,
            }.items()
            if v is not None
        }
        response = await self._post(path, body=body)
        agent = await self.retrieve_agent(response.json()["agent_id"])
        return agent

    async def delete_agent(
        self,
        agent_id: str,
    ) -> bool:
        path = f"agents/{agent_id}/delete"
        response = await self._post(path)
        return response.status_code == 200
