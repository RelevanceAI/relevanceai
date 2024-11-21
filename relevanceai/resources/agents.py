from __future__ import annotations

from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.agent import Agent
from ..types.tool import Tool
from typing import List

class Agents(SyncAPIResource): 
    
    _client: RelevanceAI

    def list_agents(
        self
    ) -> List[Agent]:
        path = "agents/list"
        response = self._post(path)
        agents = [Agent(**item) for item in response.json().get("results", [])]
        return sorted(agents, key=lambda x: (x.name is None, x.name or ""))
    
    def retrieve_agent(
        self,
        agent_id: str,
    ) -> Agent:
        path = f"agents/{agent_id}/get"
        response = self._get(path)
        return Agent(**response.json()["agent"])

    def delete_agent(
        self, 
        agent_id: str
    ) -> bool:
        path = f"agents/{agent_id}/delete"
        response = self._post(path)
        return response.status_code == 200

    def list_agent_tools(
        self,
        agent_id: str,
    ) -> List[Tool]:
        path = "agents/tools/list"
        body = {"agent_ids": [agent_id]}
        response = self._post(path, body=body)
        tools = [Tool(**item) for item in response.json().get("results", [])]
        tools = [tool for tool in tools if tool.type!='agent']
        return sorted(tools, key=lambda x: x.title or "")
    
    def list_subagents(
        self,
        agent_id: str,
    ) -> List[Tool]:
        path = "agents/tools/list"
        body = {"agent_ids": [agent_id]}
        response = self._post(path, body=body)
        tools = [Tool(**item) for item in response.json().get("results", [])]
        tools = [tool for tool in tools if tool.type=='agent']
        return sorted(tools, key=lambda x: x.title or "")