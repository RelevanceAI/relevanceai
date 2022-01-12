from relevanceai.api.client import BatchAPIClient

GROUPBY_MAPPING = {"text": "category", "numeric": "numeric"}


class Groupby(BatchAPIClient):
    def __init__(self, client, dataset_id):
        self.client = client
        self.dataset_id = dataset_id

    def __call__(self, by: list):
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
        self.agg = Agg(self.client, self.dataset_id, self.groupby_call)
        return self

    def _get_groupby_fields(self):
        """
        Get what type of groupby field to use
        """
        schema = self.client.datasets.schema(self.dataset_id)
        self._check_groupby_in_schema(schema)
        fields_schema = {k: v for k, v in schema.items() if k in self.by}
        self._check_groupby_value_type(fields_schema)

        return {k: GROUPBY_MAPPING[v] for k, v in fields_schema.items()}

    def _check_groupby_in_schema(self, schema):
        """
        Check groupby fields are in schema
        """
        invalid_fields = []
        for i in self.by:
            if i not in schema:
                invalid_fields.append(i)
        if len(invalid_fields) > 0:
            raise ValueError(
                f"{', '.join(invalid_fields)} are invalid fields. They are not in the dataset schema."
            )

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
    def __init__(self, client, dataset_id, groupby_call = []):
        self.client = client
        self.dataset_id = dataset_id
        self.groupby_call = groupby_call

    def __call__(self, metrics: dict):
        """
        Return aggregation query from metrics

        Parameters
        ----------
        metrics : dict
            Dictionary of field and metric pairs to get
   
        """
        self.metrics = metrics
        self.metrics_call = self._create_metrics()
        return self.client.services.aggregate.aggregate(
            dataset_id=self.dataset_id,
            metrics=self.metrics_call,
            groupby=self.groupby_call,
        )

    def _create_metrics(self):
        """
        Create metric call
        """
        return [{"name": k, "field": k, "agg": v} for k, v in self.metrics.items()]
