from __future__ import annotations

from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..resources.tool import Tool
from ..types.agent import *
from ..types.task import Task, TriggeredTask, ScheduledActionTrigger, TaskView
from typing import List
import json


class Agent(SyncAPIResource):
    _client: RelevanceAI

    def __init__(self, client: RelevanceAI, **metadata):
        super().__init__(client=client)
        self.metadata = AgentType(**metadata)
        self.agent_id = self.metadata.agent_id

    def list_tools(
        self,
    ) -> List[Tool]:
        path = "agents/tools/list"
        body = {"agent_ids": [self.agent_id]}
        response = self._post(path, body=body)
        tools = [Tool(**item) for item in response.json().get("results", [])]
        tools = [tool for tool in tools if tool.type != "agent"]
        return sorted(tools, key=lambda x: x.title or "")

    def list_subagents(
        self,
    ) -> List[Tool]:
        path = "agents/tools/list"
        body = {"agent_ids": [self.agent_id]}
        response = self._post(path, body=body)
        tools = [Tool(**item) for item in response.json().get("results", [])]
        tools = [tool for tool in tools if tool.type == "agent"]
        return sorted(tools, key=lambda x: x.title or "")

    def retrieve_task(self, conversation_id: str) -> Task:
        task_items = self.list_tasks(self.agent_id)
        for task_item in task_items:
            if task_item.knowledge_set == conversation_id:
                return task_item
        return task_item

    def trigger_task(self, message: str, **kwargs) -> TriggeredTask:
        path = "agents/trigger"
        body = {
            "agent_id": self.agent_id,
            "message": {
                "role": "user",
                "content": message,
            },
            **kwargs,
        }
        response = self._post(path, body=body)
        return TriggeredTask(**response.json())

    def list_tasks(
        self,
        max_results: Optional[int] = 50,
        state: Optional[str] = None,
    ) -> List[Task]:
        path = "agents/conversations/list"
        params = {
            "include_agent_details": "true",
            "include_debug_info": "false",
            "filters": json.dumps(
                [
                    {
                        "field": "conversation.is_debug_mode_task",
                        "filter_type": "exact_match",
                        "condition": "!=",
                        "condition_value": True,
                    },
                    {
                        "filter_type": "exact_match",
                        "field": "conversation.agent_id",
                        "condition_value": [self.agent_id],
                        "condition": "==",
                    },
                ]
            ),
            "sort": json.dumps([{"update_datetime": "desc"}]),
            "page_size": max_results,
        }
        response = self._get(path, params=params)
        tasks = [Task(**item) for item in response.json()["results"]]
        if state:
            tasks = [
                task
                for task in tasks
                if task.metadata.conversation.state.value == state
            ]
        return tasks

    def view_task_steps(self, conversation_id: str):
        path = f"agents/{self.agent_id}/tasks/{conversation_id}/view"
        response = self._post(path)
        return TaskView(**response.json())

    def approve_task(
        self,
        conversation_id: str,
        tool_id: str = None,
    ):
        task_view: TaskView = self.view_task_steps(self.agent_id, conversation_id)
        for t in task_view.results:
            if hasattr(t, "content") and hasattr(t.content, "requires_confirmation"):
                requires_confirmation = t.content.requires_confirmation

                if requires_confirmation and (
                    tool_id is None or t.content.tool_config.id == tool_id
                ):
                    action = t.content.action_details.action
                    action_request_id = t.content.action_details.action_request_id

                    triggered_task = self.trigger_task_from_action(
                        self.agent_id, conversation_id, action, action_request_id
                    )
                    return triggered_task

    def trigger_task_from_action(
        self, conversation_id: str, action: str, action_request_id: str
    ):
        path = f"agents/trigger"
        body = {
            "agent_id": self.agent_id,
            "conversation_id": conversation_id,
            "message": {
                "action": action,
                "action_request_id": action_request_id,
                "action_params_override": {},
                "role": "action-confirm",
            },
            "debug": True,
            "is_debug_mode_task": False,
        }
        response = self._post(path, body=body)
        return TriggeredTask(**response.json())

    def rerun_task(
        self,
        conversation_id: str,
    ) -> Optional[TriggeredTask]:
        trigger_message_data = self._get_trigger_message(self.agent_id, conversation_id)

        if not trigger_message_data:
            return None

        message_content, edit_message_id = trigger_message_data

        path = "agents/trigger"
        body = {
            "agent_id": self.agent_id,
            "conversation_id": conversation_id,
            "message": {"role": "user", "content": message_content},
            "edit_message_id": edit_message_id,
        }

        response = self._post(path, body=body)

        return TriggeredTask(**response.json())

    def _get_trigger_message(
        self,
        conversation_id: str,
    ) -> Optional[tuple[str, str]]:
        path = f"agents/{self.agent_id}/tasks/{conversation_id}/trigger_message"
        response = self._get(path)
        trigger_message_data = response.json().get("trigger_message")
        if (
            trigger_message_data
            and trigger_message_data["content"]["is_trigger_message"]
        ):
            return (
                trigger_message_data["content"]["text"],
                trigger_message_data["item_id"],
            )
        return None

    def schedule_action_in_task(
        self,
        conversation_id: str,
        message: str,
        minutes_until_schedule: int = 0,
    ) -> ScheduledActionTrigger:
        path = f"agents/{self.agent_id}/scheduled_triggers_item/create"
        body = {
            "conversation_id": conversation_id,
            "message": message,
            "minutes_until_schedule": minutes_until_schedule,
        }
        params = None
        response = self._post(path, body=body, params=params)
        return ScheduledActionTrigger(**response.json())

    def get_task_output_preview(
        self,
        conversation_id: str,
    ) -> Union[Task, bool]:
        path = f"agents/conversations/studios/list"
        params = {
            "conversation_id": conversation_id,
            "agent_id": self.agent_id,
            "page_size": 100,
        }
        response = self._get(path, params=params)
        if response.json()["results"][0]["status"] == "complete":
            return response.json()["results"][0]["output_preview"]
        return False

    def remove_all_tools(self, partial_update: Optional[bool] = True) -> None:
        path = "agents/upsert"
        self.metadata.actions = []
        body = {
            "agent_id": self.agent_id,
            "actions": [],
            "partial_update": partial_update,
        }
        response = self._post(path, body=body)
        return response.json()

    def add_tool(self, tool_id: str, partial_update: Optional[bool] = True) -> None:
        path = "agents/upsert"
        self.metadata.actions.append({"chain_id": tool_id})
        body = {
            "agent_id": self.agent_id,
            "actions": self.metadata.actions,
            "partial_update": partial_update,
        }
        response = self._post(path, body=body)
        return response.json()

    def remove_tool(self, tool_id: str, partial_update: Optional[bool] = True) -> None:
        path = "agents/upsert"
        self.metadata.actions = [
            action for action in self.metadata.actions if action["chain_id"] != tool_id
        ]
        body = {
            "agent_id": self.agent_id,
            "actions": self.metadata.actions,
            "partial_update": partial_update,
        }
        response = self._post(path, body=body)
        return response.json()
    
    # todo: approval modes for agents and tools 

    def add_subagent(self):
        # set agent settings variables 
        pass

    def remove_subagent(self): 
        pass

    #* triggers
    def add_trigger(self): 
        pass

    def remove_trigger(self):
        pass

    #* core instructions 
    def get_core_instructions(self):
        pass

    def update_core_instructions(self):
        pass

    #* template settings    
    def get_template_settings(self):
        pass

    def create_template_variable(self):
        pass

    def remove_template_variable(self):
        pass

    def update_template_settings(self):
        # partial update
        pass

    #* advanced settings 
    def get_advanced_settings(self): 
        pass 

    def update_template_settings(self):
        pass

    def get_link(self): 
        return f"https://app.relevanceai.com/agents/{self._client.region}/"


    def __repr__(self):
        return f'Agent(agent_id="{self.agent_id}", name="{self.metadata.name}")'
