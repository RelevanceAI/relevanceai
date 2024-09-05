from ..._client import RelevanceAI
from ..._resource import SyncAPIResource
from ...types.beta.agent import Agent, AgentDeleted

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
    
    def list(
        self
    ) -> List[Agent]:
        path = "agents/list"
        body = {}
        params = None
        response = self._post(path=path, body=body, params=params)
        return [Agent(**item) for item in response.json().get('results', [])]  
    
    def upsert(
        self,
    ) -> Agent:
        path = "agents/upsert"
        body = {}
        params = {}
        response = self._post(path=path, body=body, params=params)
        return self.retrieve(response.json().get("agent_id"))
    
    def delete(
        self,
        agent_id: str
    ) -> AgentDeleted:
        path = f"agents/{agent_id}/delete"
        body = None
        params = None
        response = self._post(path=path, body=body, params=params)
        if response.status_code != 200:
            return True
        else:
            return False #! raise error
        
    def cancel(
        self,
        agent_id: str
    ) -> None: 
        path = f"agents/{agent_id}/cancel"
        body = None
        params = None
        response = self._post(path=path, body=body, params=params)
        return
    
    def clone(
        self,
        template: dict,
    ) -> None:
        
        path = "agents/clone"
        body = {
            "template": template
        }
        params = None
        response = self._post(path=path, body=body, params=params)
        return
    
    def list_tools(
        self,
        agent_id: str,
    ) -> List[dict]:
        path = f"agents/{agent_id}/tools/list"
        body = None
        params = None
        response = self._post(path=path, body=body, params=params)
        return response.json().get('chains', [])
    
    
    
        
    
    
        
    
    