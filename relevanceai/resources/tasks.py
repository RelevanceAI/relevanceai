from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.task import TriggerTask, ScheduledActionTrigger, TaskItem, TaskConversation
from typing import Optional, List

class Tasks(SyncAPIResource): 

    _client: RelevanceAI
    
    def list_all_tasks(
        self,
        agent_id: str,
        max_results: Optional[int] = 50
    ) -> List[TaskItem]:
        path = "agents/conversations/list"
        params = {
            "include_agent_details": "true",
            "include_debug_info": "false",
            "filters": '[{"field":"conversation.is_debug_mode_task","filter_type":"exact_match","condition":"!=","condition_value":true},'
                       '{"filter_type":"exact_match","field":"conversation.state","condition_value":["running","starting-up"],"condition":"!="},'
                       '{"filter_type":"exact_match","field":"conversation.agent_id","condition_value":["%s"],"condition":"=="}]' % agent_id,
            "sort": '[{"update_datetime":"desc"}]',
            "page_size": max_results
        }
        response = self._get(path, params=params)
        return [TaskItem(**item) for item in response.json()['results']]

    def retrieve_task(
        self,
        agent_id: str,
        conversation_id: str
    ) -> TaskItem:
        task_items = self.list_all_tasks(agent_id)
        for task_item in task_items:
            if task_item.knowledge_set == conversation_id:
                return task_item
        return task_item

    def list_task_steps(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> TaskConversation:
        path = "agents/conversations/studios/list"
        params = {
            "agent_id": agent_id,
            "conversation_id": conversation_id,
        }
        response = self._get(path=path, params=params)
        task_conversation = TaskConversation(**response.json()["results"][0])
        task_conversation.title = self.retrieve_task(
            agent_id, conversation_id
        ).metadata.conversation.title
        return task_conversation

    def trigger_task(
        self,
        agent_id: str,
        message: str, 
    ) -> TriggerTask:
        path = "agents/trigger"
        body = {
            "agent_id": agent_id,
            "message": {
                "role": "user",
                "content": message,
            },
        }
        response = self._client.post(path, json=body)
        return TriggerTask(**response.json())

    def rerun_task(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> Optional[TriggerTask]:
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

        return TriggerTask(**response.json())

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
        conversation_id: str
    ) -> bool: 
        path = "knowledge/sets/delete"
        body = {"knowledge_set": [conversation_id]}
        response = self._post(path=path, body=body)
        return response.status_code == 200