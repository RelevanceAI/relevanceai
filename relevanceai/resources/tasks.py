from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.task import Task, TriggeredTask, ScheduledActionTrigger, TaskConversation
from typing import Optional, List
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
                {"filter_type": "exact_match", "field": "conversation.state", "condition_value": ["running", "starting-up"], "condition": "!="},
                {"filter_type": "exact_match", "field": "conversation.agent_id", "condition_value": [agent_id], "condition": "=="}
            ]),
            "sort": json.dumps([{"update_datetime": "desc"}]),
            "page_size": max_results
        }
        response = self._get(path, params=params)
        tasks = [Task(**item) for item in response.json()['results']]
        if state:
            tasks = [task for task in tasks if task.metadata.conversation.state == state]
        return tasks

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
    ) -> TriggeredTask:
        path = "agents/trigger"
        body = {
            "agent_id": agent_id,
            "message": {
                "role": "user",
                "content": message,
            },
        }
        response = self._client.post(path, json=body)
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
