import pytest
from unittest.mock import MagicMock
from relevanceai.resources.agents import AgentsManager
from relevanceai.resources.agent import Agent


class TestAgentsManager:
    @pytest.fixture
    def mock_client(self):
        """Fixture to create a mock RelevanceAI client."""
        return MagicMock()

    @pytest.fixture
    def agents_manager(self, mock_client):
        """Fixture to create an AgentsManager instance."""
        return AgentsManager(client=mock_client)

    def test_list_agents(self, agents_manager):
        """Test listing agents with successful response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"agent_id": "agent-1", "name": "Agent 1", "_id": "id-1", "project": "default"},
                {"agent_id": "agent-2", "name": None, "_id": "id-2", "project": "default"},
                {"agent_id": "agent-3", "name": "Agent 3", "_id": "id-3", "project": "default"}
            ]
        }
        agents_manager._post = MagicMock(return_value=mock_response)

        result = agents_manager.list_agents()

        agents_manager._post.assert_called_once_with("agents/list")
        assert len(result) == 3
        assert all(isinstance(agent, Agent) for agent in result)
        # Verify sorting (None names should come last)
        assert [agent.metadata.name for agent in result] == ["Agent 1", "Agent 3", None]

    def test_retrieve_agent(self, agents_manager):
        """Test retrieving a specific agent."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "agent": {
                "agent_id": "test-agent",
                "name": "Test Agent",
                "_id": "test-id",
                "project": "default"
            }
        }
        agents_manager._get = MagicMock(return_value=mock_response)

        result = agents_manager.retrieve_agent("test-agent")

        agents_manager._get.assert_called_once_with("agents/test-agent/get")
        assert isinstance(result, Agent)
        assert result.agent_id == "test-agent"
        assert result.metadata.name == "Test Agent"

    def test_upsert_agent(self, agents_manager):
        """Test upserting an agent with various parameters."""
        # Mock the POST response for upsert
        mock_upsert_response = MagicMock()
        mock_upsert_response.json.return_value = {"agent_id": "new-agent"}
        agents_manager._post = MagicMock(return_value=mock_upsert_response)

        # Mock the retrieve_agent response
        mock_agent = Agent(
            client=agents_manager._client,
            agent_id="new-agent",
            name="New Agent",
            system_prompt="Test prompt",
            model="gpt-4",
            temperature=0.7,
            _id="new-id",
            project="default"
        )
        agents_manager.retrieve_agent = MagicMock(return_value=mock_agent)

        # Test with all parameters
        result = agents_manager.upsert_agent(
            agent_id="new-agent",
            name="New Agent",
            system_prompt="Test prompt",
            model="gpt-4",
            temperature=0.7
        )

        agents_manager._post.assert_called_once_with(
            "/agents/upsert",
            body={
                "agent_id": "new-agent",
                "name": "New Agent",
                "system_prompt": "Test prompt",
                "model": "gpt-4",
                "temperature": 0.7
            }
        )
        agents_manager.retrieve_agent.assert_called_once_with("new-agent")
        assert isinstance(result, Agent)
        assert result.agent_id == "new-agent"

    def test_upsert_agent_minimal(self, agents_manager):
        """Test upserting an agent with minimal parameters."""
        mock_upsert_response = MagicMock()
        mock_upsert_response.json.return_value = {"agent_id": "new-agent"}
        agents_manager._post = MagicMock(return_value=mock_upsert_response)

        mock_agent = Agent(
            client=agents_manager._client,
            agent_id="new-agent",
            _id="new-id",
            project="default"
        )
        agents_manager.retrieve_agent = MagicMock(return_value=mock_agent)

        result = agents_manager.upsert_agent(agent_id="new-agent")

        agents_manager._post.assert_called_once_with(
            "/agents/upsert",
            body={"agent_id": "new-agent"}
        )
        assert isinstance(result, Agent)
        assert result.agent_id == "new-agent"

    def test_delete_agent(self, agents_manager):
        """Test deleting an agent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        agents_manager._post = MagicMock(return_value=mock_response)

        result = agents_manager.delete_agent("test-agent")

        agents_manager._post.assert_called_once_with("agents/test-agent/delete")
        assert result is True

    def test_delete_agent_failure(self, agents_manager):
        """Test deleting an agent with non-200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        agents_manager._post = MagicMock(return_value=mock_response)

        result = agents_manager.delete_agent("test-agent")

        agents_manager._post.assert_called_once_with("agents/test-agent/delete")
        assert result is False