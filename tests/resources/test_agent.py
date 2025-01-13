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

    def test_list_tools(self, agent):
        """Test listing tools."""
        mock_tool = MagicMock(tool_id="tool-1", title="Test Tool")
        agent.list_tools = MagicMock(return_value=[mock_tool])
        result = agent.list_tools()
        assert len(result) == 1
        assert result[0].tool_id == "tool-1"
        assert result[0].title == "Test Tool"

    @patch('relevanceai.resources.agent.Tool')
    def test_list_subagents(self, mock_tool_class, agent):
        """Test listing subagents."""
        mock_agent_tool = MagicMock(
            tool_id="agent-1",
            title="Subagent",
            metadata=MagicMock(type="agent", title="Subagent")
        )
        mock_regular_tool = MagicMock(
            tool_id="tool-1",
            title="Regular Tool",
            metadata=MagicMock(type="regular", title="Regular Tool")
        )
        
        agent._post = MagicMock(return_value=MagicMock(
            json=lambda: {"results": [
                {"tool_id": "agent-1", "type": "agent", "title": "Subagent"},
                {"tool_id": "tool-1", "type": "regular", "title": "Regular Tool"}
            ]}
        ))
        
        mock_tool_class.side_effect = [mock_agent_tool, mock_regular_tool]
        
        result = agent.list_subagents()
        
        assert len(result) == 1
        assert result[0].tool_id == "agent-1"
        assert result[0].title == "Subagent"
        
        agent._post.assert_called_once_with(
            "agents/tools/list",
            body={"agent_ids": ["test-agent"]}
        )

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

    def test_rerun_task(self, agent):
        """Test rerun task with valid trigger message."""
        agent._get_trigger_message = MagicMock(
            return_value=("Test message", "message-123")
        )
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "task_id": "task-123",
            "status": "pending"
        }
        agent._post = MagicMock(return_value=mock_response)
        
        result = agent.rerun_task("conversation-123")
        
        agent._post.assert_called_once_with(
            "agents/trigger",
            body={
                "agent_id": "test-agent",
                "conversation_id": "conversation-123",
                "message": {"role": "user", "content": "Test message"},
                "edit_message_id": "message-123"
            }
        )
        
        assert isinstance(result, TriggeredTask)
