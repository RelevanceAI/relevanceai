from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.agent import Agent

from typing import List

class Agents(SyncAPIResource): 
    
    _client: RelevanceAI
    def retrieve(
        self,
        agent_id: str,
    ) -> Agent:
        path = f"agents/{agent_id}/get"
        body = None
        params = None
        response = self._get(path=path, body=body, params=params)
        return Agent(**response.json().get('agent', []))
    
    def list_agents(
        self
    ) -> List[Agent]:
        path = "agents/list"
        body = {}
        params = None
        response = self._post(path=path, body=body, params=params)
        return [Agent(**item) for item in response.json().get('results', [])]  
    
    def delete(
        self,
        agent_id: str
    ) -> bool:
        path = f"agents/{agent_id}/delete"
        body = None
        params = None
        response = self._post(path=path, body=body, params=params)
        if response.status_code != 200:
            return True
        else:
            return False
    def list_agent_tools(
        self,
        agent_id: str,
    ) -> List[dict]:
        path = f"agents/{agent_id}/tools/list"
        body = None
        params = None
        response = self._post(path=path, body=body, params=params)
        return response.json().get('chains', [])
    
