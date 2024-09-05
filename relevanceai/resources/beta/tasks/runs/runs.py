
from .steps import Steps
from ....._resource import SyncAPIResource

#! CALL THESE TRIGGERS INSTEAD?

class Runs(SyncAPIResource):

    def steps(self) -> Steps: 
        
        return Steps(self._client)