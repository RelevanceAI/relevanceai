import pytest
from unittest.mock import Mock, patch
from ..types.knowledge import KnowledgeSet, KnowledgeRow
from ..resources.knowledge import Knowledge

@pytest.fixture
def knowledge():
    client_mock = Mock()
    return Knowledge(client_mock)

def test_list_knowledge_success(knowledge):
    response_mock = Mock()
    response_mock.json.return_value = {
        "results": [
            {"id": "set1", "name": "Knowledge Set 1"},
            {"id": "set2", "name": "Knowledge Set 2"}
        ]
    }
    knowledge._post.return_value = response_mock

    knowledge_sets = knowledge.list_knowledge()
    assert len(knowledge_sets) == 2
    assert isinstance(knowledge_sets[0], KnowledgeSet)
    assert knowledge_sets[0].id == "set1"
    assert knowledge_sets[1].id == "set2"

def test_retrieve_knowledge_success(knowledge):
    response_mock = Mock()
    response_mock.json.return_value = {
        "results": [
            {"id": "row1", "data": "Knowledge Data 1"},
            {"id": "row2", "data": "Knowledge Data 2"}
        ]
    }
    knowledge._post.return_value = response_mock

    knowledge_rows = knowledge.retrieve_knowledge("set1")
    assert len(knowledge_rows) == 2
    assert isinstance(knowledge_rows[0], KnowledgeRow)
    assert knowledge_rows[0].id == "row1"
    assert knowledge_rows[1].id == "row2"

def test_delete_knowledge_success(knowledge):
    response_mock = Mock()
    response_mock.status_code = 200
    knowledge._post.return_value = response_mock

    result = knowledge.delete_knowledge("set1")
    assert result is True

def test_delete_knowledge_failed(knowledge):
    response_mock = Mock()
    response_mock.status_code = 400
    knowledge._post.return_value = response_mock

    result = knowledge.delete_knowledge("set1")
    assert result is False
