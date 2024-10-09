
from __future__ import annotations

import os 
from typing import TYPE_CHECKING
import httpx

if TYPE_CHECKING:
    from . import resources

from ._base_client import SyncAPIClient 

class RelevanceAI(SyncAPIClient): 
    
    agents: resources.Agents
    tasks: resources.Tasks
    tools: resources.Tools
    knowledge: resources.Knowledge
    
    api_key: str
    region: str | None
    project: str | None
    
    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: str | None = None,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
    ) -> None:
        
        if api_key is None: 
            api_key = os.environ.get("RAI_API_KEY")
        if api_key is None: 
            raise ValueError("API key is required")
        self.api_key = api_key
        
        if region is None: 
            region = os.environ.get("RAI_REGION")
        if region is None: 
            raise ValueError("Region is required")
        self.region = region
        
        if project is None: 
            project = os.environ.get("RAI_PROJECT")
        if project is None: 
            raise ValueError("Project is required")
        self.project = project
        
        headers = {"Authorization": f"{self.project}:{self.api_key}"}
        base_url = f"https://api-{self.region}.stack.tryrelevance.com/latest"

        super().__init__(base_url=base_url, headers=headers)
        
        from . import resources
        self.agents = resources.Agents(self)
        self.tasks = resources.Tasks(self)
        self.tools = resources.Tools(self)
        self.knowledge = resources.Knowledge(self)

    
if __name__=="__main__": 
    
    client = RelevanceAI()
    
