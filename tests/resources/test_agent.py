import pytest
from unittest.mock import MagicMock, patch
from relevanceai.resources.agent import Agent
from relevanceai.types.agent import AgentType
from relevanceai.resources.tool import Tool
from relevanceai.types.task import Task, TriggeredTask, ScheduledActionTrigger, TaskView


class TestAgent:
    @pytest.fixture
    def mock_client(self):
        """Fixture to create a mock RelevanceAI client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def agent(self, mock_client):
        """Fixture to create an Agent instance."""
        metadata = {
            "agent_id": "test-agent",
            "name": "Test Agent",
            "_id": "agent-id-123",
            "project": "default-project",
        }
        return Agent(client=mock_client, **metadata)

    def test_agent_init(self, agent):
        """Test initialization of the Agent class."""
        assert agent.agent_id == "test-agent"
        assert agent.metadata.name == "Test Agent"

    def test_delete_agent(self, agent):
        """Test deleting an agent."""
        agent._post = MagicMock(return_value=MagicMock(status_code=200))
        result = agent.delete_agent()
        agent._post.assert_called_once_with("agents/test-agent/delete")
        assert result is True

    def test_retrieve_task(self, agent):
        """Test retrieving a task by conversation ID."""
        task_mock = MagicMock(knowledge_set="conversation-1")
        agent.list_tasks = MagicMock(return_value=[task_mock])
        task = agent.retrieve_task("conversation-1")
        assert task.knowledge_set == "conversation-1"

    def test_approve_task(self, agent):
        """Test approving a task."""
        task_view_mock = MagicMock(results=[MagicMock(content=MagicMock(
            requires_confirmation=True,
            tool_config=MagicMock(id="tool-1"),
            action_details=MagicMock(action="action-1", action_request_id="req-1"),
        ))])
        agent.view_task_steps = MagicMock(return_value=task_view_mock)
        agent.trigger_task_from_action = MagicMock(return_value="Triggered")
        result = agent.approve_task("conversation-1", tool_id="tool-1")
        assert result == "Triggered"