from relevanceai import _client
from functools import cached_property

from ..._client import RelevanceAI
from .tasks import Tasks
from .agents import Agents

class Beta():
    
    _client: RelevanceAI 
    
    def __init__(self, client=None):
        self._client = client
    
    @cached_property
    def agents(self) -> Agents:
        return Agents(self._client)
    
    @cached_property
    def tasks(self) -> Tasks:
        return Tasks(self._client)
    
        
    
