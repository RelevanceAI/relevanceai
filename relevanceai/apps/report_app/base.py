class ReportBase:
    def __init__(self, name: str, dataset, deployable_id: str = None):
        self.name = name
        self.dataset = dataset
        self.dataset_id = dataset.dataset_id
        self.deployable_id = deployable_id
        app_config = None
        self.reloaded = False
        if deployable_id:
            try:
                app_config = self.dataset.get_app(deployable_id)
                self.reloaded = True
            except:
                raise Exception(
                    f"{deployable_id} does not exist in the dataset, the given id will be used for creating a new app."
                )
        if app_config:
            self.config = app_config["configuration"]
        else:
            self.config = {
                "dataset_name": self.dataset_id,
                "deployable_name": self.name,
                "type": "page",
                "page-content": {"type": "doc", "content": []},
            }

    @property
    def contents(self):
        return self.config["page-content"]["content"]

    def deploy(self, overwrite: bool = False):
        if self.deployable_id and self.reloaded:
            status = self.dataset.update_app(
                self.deployable_id, self.config, overwrite=overwrite
            )
            if status["status"] == "success":
                return self.dataset.get_app(self.deployable_id)
            else:
                raise Exception("Failed to update app")
        return self.dataset.create_app(self.config)
