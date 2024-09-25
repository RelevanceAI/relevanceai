import pytest
from unittest.mock import Mock, patch
from ..types.task import TriggerTask, ScheduledActionTrigger, TaskItem, TaskConversation
from ..resources.tasks import Tasks

@pytest.fixture
def tasks():
    client_mock = Mock()
    return Tasks(client_mock)

def test_list_tasks_success(tasks):
    response_mock = Mock()
    response_mock.json.return_value = {
        "results": [
            {"id": "task1", "metadata": {"conversation": {"state": "active"}}},
            {"id": "task2", "metadata": {"conversation": {"state": "completed"}}}
        ]
    }
    tasks._get.return_value = response_mock

    task_list = tasks.list_tasks("agent1", max_results=10, state="active")
    assert len(task_list) == 1
    assert isinstance(task_list[0], TaskItem)
    assert task_list[0].metadata.conversation.state == "active"

def test_retrieve_task_success(tasks):
    task_mock = Mock()
    task_mock.metadata.conversation.state = "active"
    task_mock.knowledge_set = "conversation1"
    tasks.list_all_tasks = Mock(return_value=[task_mock])

    retrieved_task = tasks.retrieve_task("agent1", "conversation1")
    assert isinstance(retrieved_task, TaskItem)
    assert retrieved_task.knowledge_set == "conversation1"

def test_list_task_steps_success(tasks):
    response_mock = Mock()
    response_mock.json.return_value = {
        "results": [{"title": "Task Conversation", "steps": []}]
    }
    tasks._get.return_value = response_mock
    tasks.retrieve_task = Mock(return_value=Mock(metadata=Mock(conversation=Mock(title="Task Conversation Title"))))

    task_steps = tasks.list_task_steps("agent1", "conversation1")
    assert isinstance(task_steps, TaskConversation)
    assert task_steps.title == "Task Conversation Title"

def test_trigger_task_success(tasks):
    response_mock = Mock()
    response_mock.json.return_value = {
        "id": "triggered_task",
        "status": "success"
    }
    tasks._client.post.return_value = response_mock

    triggered_task = tasks.trigger_task("agent1", "Test message")
    assert isinstance(triggered_task, TriggerTask)
    assert triggered_task.id == "triggered_task"
    assert triggered_task.status == "success"

def test_rerun_task_success(tasks):
    response_mock = Mock()
    response_mock.json.return_value = {
        "id": "rerun_task",
        "status": "success"
    }
    tasks._get_trigger_message = Mock(return_value=("previous_message", "edit_message_id"))
    tasks._post.return_value = response_mock

    rerun_task = tasks.rerun_task("agent1", "conversation1")
    assert isinstance(rerun_task, TriggerTask)
    assert rerun_task.id == "rerun_task"
    assert rerun_task.status == "success"

def test_rerun_task_no_trigger_message(tasks):
    tasks._get_trigger_message = Mock(return_value=None)

    rerun_task = tasks.rerun_task("agent1", "conversation1")
    assert rerun_task is None

def test_schedule_action_in_task_success(tasks):
    response_mock = Mock()
    response_mock.json.return_value = {
        "scheduled_trigger_id": "trigger1",
        "status": "scheduled"
    }
    tasks._post.return_value = response_mock

    scheduled_action = tasks.schedule_action_in_task("agent1", "conversation1", "Schedule this action", 10)
    assert isinstance(scheduled_action, ScheduledActionTrigger)
    assert scheduled_action.scheduled_trigger_id == "trigger1"
    assert scheduled_action.status == "scheduled"

def test_delete_task_success(tasks):
    response_mock = Mock()
    response_mock.status_code = 200
    tasks._post.return_value = response_mock

    result = tasks.delete_task("conversation1")
    assert result is True

def test_delete_task_failed(tasks):
    response_mock = Mock()
    response_mock.status_code = 400
    tasks._post.return_value = response_mock

    result = tasks.delete_task("conversation1")
    assert result is False
