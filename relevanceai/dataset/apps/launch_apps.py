from relevanceai.dataset.apps.templates import CreateAppsTemplates
from relevanceai.utils.decorators.analytics import track
from relevanceai.constants import EXPLORER_APP_LINK

class LaunchApps(CreateAppsTemplates):
    """
    Launch apps are designed to create apps automatically.
    """
    @track
    def launch_explore_app(self):
        print(EXPLORER_APP_LINK.format(self.dataset_id))

    @track
    def launch_search_app(self):
        print(EXPLORER_APP_LINK.format(self.dataset_id))