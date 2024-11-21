from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.task import Task, TriggeredTask, ScheduledActionTrigger, TaskView, TaskStep
from ..types.knowledge import Metadata
from typing import Optional, List, Union
import json 

class Tasks(SyncAPIResource): 

    _client: RelevanceAI
    
    def list_tasks( 
        self,
        agent_id: str,
        max_results: Optional[int] = 50,
        state: Optional[str] = None,
    ) -> List[Task]:
        path = "agents/conversations/list"
        params = {
            "include_agent_details": "true",
            "include_debug_info": "false",
            "filters": json.dumps([
                {"field": "conversation.is_debug_mode_task", "filter_type": "exact_match", "condition": "!=", "condition_value": True},
                {"filter_type": "exact_match", "field": "conversation.agent_id", "condition_value": [agent_id], "condition": "=="}
            ]),
            "sort": json.dumps([{"update_datetime": "desc"}]),
            "page_size": max_results
        }
        response = self._get(path, params=params)
        tasks = [Task(**item) for item in response.json()['results']]
        if state:
            tasks = [task for task in tasks if task.metadata.conversation.state.value == state]
        return tasks
    
    def view_task_steps(
        self, 
        agent_id: str, 
        conversation_id: str
    ) : 
        path = f"agents/{agent_id}/tasks/{conversation_id}/view"
        response = self._post(path)
        return TaskView(**response.json())
    
    def approve_task(
        self, 
        agent_id: str, 
        conversation_id: str, 
        tool_id: str = None, 
    ): 
        task_view: TaskView = self.view_task_steps(agent_id, conversation_id)
        for t in task_view.results:
            if hasattr(t, 'content') and hasattr(t.content, 'requires_confirmation'):
                requires_confirmation = t.content.requires_confirmation
                
                if requires_confirmation and (tool_id is None or t.content.tool_config.id == tool_id):
                    action = t.content.action_details.action
                    action_request_id = t.content.action_details.action_request_id
                    
                    triggered_task = self.trigger_task_from_action(agent_id, conversation_id, action, action_request_id)
                    return triggered_task

    def retrieve_task(
        self,
        agent_id: str,
        conversation_id: str
    ) -> Task:
        task_items = self.list_tasks(agent_id)
        for task_item in task_items:
            if task_item.knowledge_set == conversation_id:
                return task_item
        return task_item

    def trigger_task(
        self,
        agent_id: str,
        message: str, 
        override: bool = False,
        debug_mode_config_id: dict | str = "default", 
    ) -> TriggeredTask:
        path = "agents/trigger"
        body = {
            "agent_id": agent_id,
            "message": {
                "role": "user",
                "content": message,
            }
        }
        if override:
            body.update({
                "debug": True,
                "is_debug_mode_task": True,
                "debug_mode_config_id": debug_mode_config_id
            })
        response = self._post(path, body=body)
        return TriggeredTask(**response.json())
    
    def trigger_task_from_action(
        self,
        agent_id: str, 
        conversation_id: str, 
        action: str, 
        action_request_id: str
    ): 
        path = f"agents/trigger"
        body = { 
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "message": {
                "action": action,
                "action_request_id": action_request_id,
                "action_params_override": {},
                "role": "action-confirm"
            },
            "debug": True,
            "is_debug_mode_task": False
        }
        response = self._post(path, body=body)
        return TriggeredTask(**response.json())

    def rerun_task(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> Optional[TriggeredTask]:
        trigger_message_data = self._get_trigger_message(agent_id, conversation_id)

        if not trigger_message_data:
            return None

        message_content, edit_message_id = trigger_message_data

        path = "agents/trigger"
        body = {
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "message": {"role": "user", "content": message_content},
            "edit_message_id": edit_message_id,
        }

        response = self._post(path, body=body)

        return TriggeredTask(**response.json())

    def _get_trigger_message(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> Optional[tuple[str, str]]:
        path = f"agents/{agent_id}/tasks/{conversation_id}/trigger_message"
        response = self._get(path)
        trigger_message_data = response.json().get("trigger_message")
        if trigger_message_data and trigger_message_data["content"]["is_trigger_message"]:
            return (
                trigger_message_data["content"]["text"],
                trigger_message_data["item_id"],
            )
        return None

    def schedule_action_in_task(
        self,
        agent_id: str,
        conversation_id: str,
        message: str, 
        minutes_until_schedule: int = 0,
    ) -> ScheduledActionTrigger: 
        path = f"agents/{agent_id}/scheduled_triggers_item/create" 
        body = { 
            "conversation_id": conversation_id,
            "message": message,
            "minutes_until_schedule": minutes_until_schedule
        }
        params = None
        response = self._post(path, body=body, params=params)
        return ScheduledActionTrigger(**response.json())

    def delete_task(
        self,
        agent_id: str,
        conversation_id: str
    ) -> bool: 
        path = "knowledge/sets/delete"
        body = {"knowledge_set": [conversation_id]}
        response = self._post(path=path, body=body)
        return response.status_code == 200

    def get_metadata(
            self, 
            conversation_id: str
    ) -> Union[Metadata, dict]:
        path = f"knowledge/sets/{conversation_id}/get_metadata"
        response = self._get(path).json()
        return Metadata(**response['metadata']) if response.get('metadata') else {}
    
    def get_task_output_preview(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> Task | bool: 
        path = f"agents/conversations/studios/list"
        params = {
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "page_size": 100
        }
        response = self._get(path, params=params)
        if response.json()["results"][0]["status"] == "complete":
            return response.json()["results"][0]["output_preview"]
        return False
