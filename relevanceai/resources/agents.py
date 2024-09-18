from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.agent import Agent
from typing import List

class Agents(SyncAPIResource): 
    
    _client: RelevanceAI

    def retrieve_agent(
        self,
        agent_id: str,
    ) -> Agent:
        path = f"agents/{agent_id}/get"
        response = self._get(path)
        return Agent(**response.json()["agent"])
    
    def list_agents(
        self
    ) -> List[Agent]:
        response = self._post("agents/list")
        return [Agent(**item) for item in response.json().get("results", [])]
    
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
    ) -> List[dict]:
        path = f"agents/{agent_id}/tools/list"
        body = None
        params = None
        response = self._post(path=path, body=body, params=params)
        return response.json().get('chains', [])
    
    def list_subagents(
        self,
    ): 
        pass