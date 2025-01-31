from __future__ import annotations

import json
import uuid
from typing import List

from .._client import RelevanceAI, AsyncRelevanceAI
from .._resource import SyncAPIResource, AsyncAPIResource
from ..resources.tool import Tool
from ..types.agent import *
from ..types.params import *
from ..types.task import Task, TriggeredTask, ScheduledActionTrigger, TaskView
from ..types.oauth import ActiveIntegrations


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
        tools = [
            Tool(client=self._client, **item)
            for item in response.json().get("results", [])
        ]
        tools = [tool for tool in tools if tool.metadata.type != "agent"]
        return sorted(tools, key=lambda x: x.metadata.title or "")

    def list_subagents(
        self,
    ) -> List[Tool]:
        path = "agents/tools/list"
        body = {"agent_ids": [self.agent_id]}
        response = self._post(path, body=body)
        tools = [
            Tool(client=self._client, **item)
            for item in response.json().get("results", [])
        ]
        tools = [tool for tool in tools if tool.metadata.type == "agent"]
        return sorted(tools, key=lambda x: x.metadata.title or "")

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
        task_view: TaskView = self.view_task_steps(conversation_id)
        for t in task_view.results:
            if hasattr(t, "content") and hasattr(t.content, "requires_confirmation"):
                requires_confirmation = t.content.requires_confirmation

                if requires_confirmation and (
                    tool_id is None or t.content.tool_config.id == tool_id
                ):
                    action = t.content.action_details.action
                    action_request_id = t.content.action_details.action_request_id

                    triggered_task = self.trigger_task_from_action(
                        conversation_id, action, action_request_id
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
        trigger_message_data = self._get_trigger_message(conversation_id)

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

    def add_subagent(
        self,
        agent_id: str,
        partial_update: Optional[bool] = True,
        action_behaviour: str = "always-ask",  # 'never-ask' | 'agent-decide'
    ):
        path = "agents/upsert"
        subagent = self._client.agents.retrieve_agent(agent_id=agent_id)
        if self.metadata.actions is None:
            self.metadata.actions = []
        self.metadata.actions.append(
            {
                "action_behaviour": action_behaviour,
                "agent_id": subagent.metadata.agent_id,
                "default_values": {},
                "title": subagent.metadata.name,
            }
        )
        body = {
            "agent_id": self.agent_id,
            "actions": self.metadata.actions,
            "partial_update": partial_update,
        }
        response = self._post(path, body=body)
        return response.json()

    def remove_subagent(self, agent_id: str, partial_update: Optional[bool] = True):
        path = "agents/upsert"
        self.metadata.actions = [
            action
            for action in self.metadata.actions
            if action.get("agent_id") != agent_id
        ]
        body = {
            "agent_id": self.agent_id,
            "actions": self.metadata.actions,
            "partial_update": partial_update,
        }
        response = self._post(path, body=body)
        return response.json()

    def update_core_instructions(
        self, core_instructions: str, partial_update: Optional[bool] = True
    ):
        path = "agents/upsert"
        body = {
            "agent_id": self.agent_id,
            "system_prompt": core_instructions,
            "partial_update": partial_update,
        }
        response = self._post(path, body=body)
        return response.json()

    def update_template_settings(
        self, params: Dict[str, ParamsBase], partial_update: Optional[bool] = True
    ):
        params_schema = {
            "properties": {},
            "required": [],
        }

        param_values = {
            field_name: param.value
            for field_name, param in params.items()
            if param.value is not None
        }

        for field_name, param in params.items():
            param_dict = param.model_dump(exclude_none=True)
            param_dict.pop("required", None)
            params_schema["properties"][field_name] = param_dict
            if param.required:
                params_schema["required"].append(field_name)

        path = "agents/upsert"
        body = {
            "agent_id": self.agent_id,
            "params": param_values,
            "params_schema": params_schema,
            "partial_update": partial_update,
        }
        response = self._post(path, body=body)
        return response.json()

    def add_google_trigger(self, google_integration_object: ActiveIntegrations) -> None:
        document_id = str(uuid.uuid4())
        path = f"syncs/items/{document_id}/upsert"
        oath_account_id = google_integration_object.account_id
        oath_label = google_integration_object.label
        body = {
            "data": {
                "destination": {"agent_id": self.agent_id},
                "config": {
                    "type": "gmail",
                    "gmail": {
                        "oauth_account_id": oath_account_id,
                        "oauth_account_label": oath_label,
                        "include_labels": [],
                        "exclude_emails": [],
                        "labels": [],
                    },
                },
                "state": {"status": "in_progress"},
                "contract": {},
            }
        }

        response = self._post(path, body=body).json()
        return {"result": "Success!"} if response == {} else response

    # todo: triggers, abilities, and advanced settings

    def get_link(self):
        return f"https://app.relevanceai.com/agents/{self._client.region}/{self._client.project}/{self.agent_id}"

    def __repr__(self):
        return f'Agent(agent_id="{self.agent_id}", name="{self.metadata.name}")'


class AsyncAgent(AsyncAPIResource):
    _client: AsyncRelevanceAI

    def __init__(self, client: RelevanceAI, **metadata):
        super().__init__(client=client)
        self.metadata = AgentType(**metadata)
        self.agent_id = self.metadata.agent_id

    async def list_tools(
        self,
    ) -> List[Tool]:
        path = "agents/tools/list"
        body = {"agent_ids": [self.agent_id]}
        response = await self._post(path, body=body)
        tools = [
            Tool(client=self._client, **item)
            for item in response.json().get("results", [])
        ]
        tools = [tool for tool in tools if tool.metadata.type != "agent"]
        return sorted(tools, key=lambda x: x.metadata.title or "")

    async def list_subagents(
        self,
    ) -> List[Tool]:
        path = "agents/tools/list"
        body = {"agent_ids": [self.agent_id]}
        response = await self._post(path, body=body)
        tools = [
            Tool(client=self._client, **item)
            for item in response.json().get("results", [])
        ]
        tools = [tool for tool in tools if tool.metadata.type == "agent"]
        return sorted(tools, key=lambda x: x.metadata.title or "")

    async def retrieve_task(self, conversation_id: str) -> Task:
        task_items = await self.list_tasks(self.agent_id)
        for task_item in task_items:
            if task_item.knowledge_set == conversation_id:
                return task_item
        return task_item

    async def trigger_task(self, message: str, **kwargs) -> TriggeredTask:
        path = "agents/trigger"
        body = {
            "agent_id": self.agent_id,
            "message": {
                "role": "user",
                "content": message,
            },
            **kwargs,
        }
        response = await self._post(path, body=body)
        return TriggeredTask(**response.json())

    async def list_tasks(
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
        response = await self._get(path, params=params)
        tasks = [Task(**item) for item in response.json()["results"]]
        if state:
            tasks = [
                task
                for task in tasks
                if task.metadata.conversation.state.value == state
            ]
        return tasks

    async def view_task_steps(self, conversation_id: str):
        path = f"agents/{self.agent_id}/tasks/{conversation_id}/view"
        response = await self._post(path)
        return TaskView(**response.json())

    async def approve_task(
        self,
        conversation_id: str,
        tool_id: str = None,
    ):
        task_view: TaskView = await self.view_task_steps(conversation_id)
        for t in task_view.results:
            if hasattr(t, "content") and hasattr(t.content, "requires_confirmation"):
                requires_confirmation = t.content.requires_confirmation

                if requires_confirmation and (
                    tool_id is None or t.content.tool_config.id == tool_id
                ):
                    action = t.content.action_details.action
                    action_request_id = t.content.action_details.action_request_id

                    triggered_task = await self.trigger_task_from_action(
                        conversation_id, action, action_request_id
                    )
                    return triggered_task

    async def trigger_task_from_action(
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
        response = await self._post(path, body=body)
        return TriggeredTask(**response.json())

    async def rerun_task(
        self,
        conversation_id: str,
    ) -> Optional[TriggeredTask]:
        trigger_message_data = await self._get_trigger_message(conversation_id)

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

        response = await self._post(path, body=body)
        return TriggeredTask(**response.json())

    async def _get_trigger_message(
        self,
        conversation_id: str,
    ) -> Optional[tuple[str, str]]:
        path = f"agents/{self.agent_id}/tasks/{conversation_id}/trigger_message"
        response = await self._get(path)
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

    async def schedule_action_in_task(
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
        response = await self._post(path, body=body, params=params)
        return ScheduledActionTrigger(**response.json())

    async def get_task_output_preview(
        self,
        conversation_id: str,
    ) -> Union[Task, bool]:
        path = f"agents/conversations/studios/list"
        params = {
            "conversation_id": conversation_id,
            "agent_id": self.agent_id,
            "page_size": 100,
        }
        response = await self._get(path, params=params)
        if response.json()["results"][0]["status"] == "complete":
            return response.json()["results"][0]["output_preview"]
        return False

    async def remove_all_tools(self, partial_update: Optional[bool] = True) -> None:
        path = "agents/upsert"
        self.metadata.actions = []
        body = {
            "agent_id": self.agent_id,
            "actions": [],
            "partial_update": partial_update,
        }
        response = await self._post(path, body=body)
        return response.json()

    async def add_tool(
        self, tool_id: str, partial_update: Optional[bool] = True
    ) -> None:
        path = "agents/upsert"
        self.metadata.actions.append({"chain_id": tool_id})
        body = {
            "agent_id": self.agent_id,
            "actions": self.metadata.actions,
            "partial_update": partial_update,
        }
        response = await self._post(path, body=body)
        return response.json()

    async def remove_tool(
        self, tool_id: str, partial_update: Optional[bool] = True
    ) -> None:
        path = "agents/upsert"
        self.metadata.actions = [
            action for action in self.metadata.actions if action["chain_id"] != tool_id
        ]
        body = {
            "agent_id": self.agent_id,
            "actions": self.metadata.actions,
            "partial_update": partial_update,
        }
        response = await self._post(path, body=body)
        return response.json()

    async def add_subagent(
        self,
        agent_id: str,
        partial_update: Optional[bool] = True,
        action_behaviour: str = "always-ask",  # 'never-ask' | 'agent-decide'
    ):
        path = "agents/upsert"
        subagent = await self._client.agents.retrieve_agent(agent_id=agent_id)
        if self.metadata.actions is None:
            self.metadata.actions = []
        self.metadata.actions.append(
            {
                "action_behaviour": action_behaviour,
                "agent_id": subagent.metadata.agent_id,
                "default_values": {},
                "title": subagent.metadata.name,
            }
        )
        body = {
            "agent_id": self.agent_id,
            "actions": self.metadata.actions,
            "partial_update": partial_update,
        }
        response = await self._post(path, body=body)
        return response.json()

    async def remove_subagent(
        self, agent_id: str, partial_update: Optional[bool] = True
    ):
        path = "agents/upsert"
        self.metadata.actions = [
            action
            for action in self.metadata.actions
            if action.get("agent_id") != agent_id
        ]
        body = {
            "agent_id": self.agent_id,
            "actions": self.metadata.actions,
            "partial_update": partial_update,
        }
        response = await self._post(path, body=body)
        return response.json()

    async def update_core_instructions(
        self, core_instructions: str, partial_update: Optional[bool] = True
    ):
        path = "agents/upsert"
        body = {
            "agent_id": self.agent_id,
            "system_prompt": core_instructions,
            "partial_update": partial_update,
        }
        response = await self._post(path, body=body)
        return response.json()

    async def update_template_settings(
        self, params: Dict[str, ParamsBase], partial_update: Optional[bool] = True
    ):
        params_schema = {
            "properties": {},
            "required": [],
        }

        param_values = {
            field_name: param.value
            for field_name, param in params.items()
            if param.value is not None
        }

        for field_name, param in params.items():
            param_dict = param.model_dump(exclude_none=True)
            param_dict.pop("required", None)
            params_schema["properties"][field_name] = param_dict
            if param.required:
                params_schema["required"].append(field_name)

        path = "agents/upsert"
        body = {
            "agent_id": self.agent_id,
            "params": param_values,
            "params_schema": params_schema,
            "partial_update": partial_update,
        }
        response = await self._post(path, body=body)
        return response.json()

    def get_link(self):
        return f"https://app.relevanceai.com/agents/{self._client.region}/{self._client.project}/{self.agent_id}"

    def __repr__(self):
        return f'AsyncAgent(agent_id="{self.agent_id}", name="{self.metadata.name}")'
