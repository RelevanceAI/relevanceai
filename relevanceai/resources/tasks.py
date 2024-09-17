
from .._client import RelevanceAI
from ..types.task import Task
from .._resource import SyncAPIResource

class Tasks(SyncAPIResource): 

    _client: RelevanceAI
    
    def trigger_task(
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
    
    def schedule_task(
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

    def list_conversations(
        self,
        agent_id: str
    ):
        path = "agents/conversations/list"
        params = {}
        response = self._get(path=path, params=params)
        filtered_conversations = [item for item in response.json()['results'] if item['metadata']['conversation']['agent_id'] == agent_id]
        return filtered_conversations

    def list_conversation_steps(
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
    
    def delete_conversation(
        self,
        conversation_id: str
    ): 
        path = "knowledge/sets/delete"
        body = {"knowledge_set": [conversation_id]}
        response = self._post(path=path, body=body)
        return response.json()

    
    


        
        
        
        
    

        

        
        