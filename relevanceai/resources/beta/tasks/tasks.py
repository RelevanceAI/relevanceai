

from ....types.beta.tasks.trigger import Trigger
from ....types.beta.task import Task
from ...._resource import SyncAPIResource
from .runs import Runs
from .messages import Messages
from functools import cached_property


class Tasks(SyncAPIResource): 
    
    @cached_property
    def runs(self) -> Runs: 
        return Runs(self._client)
    @cached_property
    def messages(self) -> Messages:
        return Messages(self._client)
    
    def trigger(
        self,
        message: str, 
        agent_id,
    ) -> Task: 
        
        path = "agents/trigger"
        body = {
            "agent_id": agent_id,
            "message": message,
        }
        response = self._post(path, body=body)
        return Task(**response.json())
    
    def list_conversation(
        self, 
        agent_id: str,
        conversation_id: str,
    ): 
        path = f"agents/conversations/studios/list"
        body = None
        params = {
            "agent_id": agent_id,
            "conversation_id": conversation_id
        }
        response = self._get(path=path, body=body, params=params)
        return response.json().get('results', [])
    
    def schedule_trigger(
        self,
        message: str, 
        conversation_id: str,
        minutes_until_schedule: int = 0,
    ) -> Trigger: 
        
        pass

        
        
        
        
    

        

        
        