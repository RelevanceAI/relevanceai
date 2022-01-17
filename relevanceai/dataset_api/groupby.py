from relevanceai.api.client import BatchAPIClient

GROUPBY_MAPPING = {"text": "category", "numeric": "numeric"}


class Groupby(BatchAPIClient):
    def __init__(self, project, api_key, dataset_id, _pre_groupby=None):
        self.project = project
        self.api_key = api_key
        self.dataset_id = dataset_id
        self._pre_groupby = _pre_groupby
        super().__init__(project=project, api_key=api_key)

    def __call__(self, by: list = []):
        """
        Instaniates Groupby Class which stores a groupby call

        Parameters
        ----------
        by : list
            List of fields to groupby

        """
        self.by = by
        self.groupby_fields = self._get_groupby_fields()
        self.groupby_call = self._create_groupby_call()
        if self._pre_groupby is not None:
            self.groupby_call += self._pre_groupby
        self.agg = Agg(self.project, self.api_key, self.dataset_id, self.groupby_call)
        return self

    def _get_groupby_fields(self):
        """
        Get what type of groupby field to use
        """
        schema = self.datasets.schema(self.dataset_id)
        self._are_fields_in_schema(self.by, self.dataset_id, schema)
        fields_schema = {k: v for k, v in schema.items() if k in self.by}
        self._check_groupby_value_type(fields_schema)

        return {k: GROUPBY_MAPPING[v] for k, v in fields_schema.items()}

    def _check_groupby_value_type(self, fields_schema):
        """
        Check groupby fields can be grouped
        """
        invalid_fields = []
        for k, v in fields_schema.items():
            if isinstance(v, dict):
                invalid_fields.append(k)
            elif v not in GROUPBY_MAPPING:
                invalid_fields.append(k)
        if len(invalid_fields) > 0:
            raise ValueError(
                f"{', '.join(invalid_fields)} are invalid fields. Groupby fields must be {' or '.join(GROUPBY_MAPPING.keys())}."
            )
        return

    def _create_groupby_call(self):
        """
        Create groupby call
        """
        return [
            {"name": k, "field": k, "agg": v} for k, v in self.groupby_fields.items()
        ]


class Agg(BatchAPIClient):
    def __init__(self, project, api_key, dataset_id, groupby_call=[]):
        self.project = project
        self.api_key = api_key
        self.dataset_id = dataset_id
        self.groupby_call = groupby_call
        super().__init__(project=project, api_key=api_key)

    def __call__(
        self,
        metrics: dict = {},
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
        alias: str = "default",
    ):
        """
        Return aggregation query from metrics

        Parameters
        ----------
        metrics : dict
            Dictionary of field and metric pairs to get
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        asc: bool
            Whether to sort results by ascending or descending order
        flatten: bool
            Whether to flatten
        alias: string
            Alias used to name a vector field. Belongs in field_{alias} vector
        """
        self.metrics = metrics
        self._are_fields_in_schema(self.metrics.keys(), self.dataset_id)
        self.metrics_call = self._create_metrics()
        return self.services.aggregate.aggregate(
            dataset_id=self.dataset_id,
            metrics=self.metrics_call,
            groupby=self.groupby_call,
            page_size=page_size,
            page=page,
            asc=asc,
            flatten=flatten,
            alias=alias,
        )

    def _create_metrics(self):
        """
        Create metric call
        """
        return [{"name": k, "field": k, "agg": v} for k, v in self.metrics.items()]
