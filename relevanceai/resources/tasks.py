
from ..types.task import Task
from .._resource import SyncAPIResource

class Tasks(SyncAPIResource): 

    # todo: set template settings for task runs 
    
    def trigger(
        self,
        message: str, 
        agent_id,
    ) -> Task: 
        path = "agents/trigger"
        body = {
            "agent_id": agent_id,
            "message": {
                "role": "user",
                "content": message,
            }
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
        agent_id: str,
        message: str, 
        conversation_id: str,
        minutes_until_schedule: int = 0,
    ): 
        path = f"agents/{agent_id}/scheduled_triggers_item/create" 
        body = { 
            "conversation_id": conversation_id,
            "message": message,
            "minutes_until_schedule": minutes_until_schedule
        }
        params = None
        response = self._post(path, body=body, params=params)
        return response.json()


        
        
        
        
    

        

        
        