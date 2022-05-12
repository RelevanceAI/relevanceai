from typing import Optional

from relevanceai.client.helpers import Credentials
from relevanceai.operations.cluster.constants import GROUPBY_MAPPING
from relevanceai._api import APIClient


class ClusterGroupby(APIClient):
    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
        alias: str,
        vector_fields: Optional[list] = None,
        _pre_groupby=None,
    ):
        self.credentials = credentials
        self.dataset_id = dataset_id
        self._pre_groupby = _pre_groupby
        self.alias = alias
        self.vector_fields = [] if vector_fields is None else vector_fields

    def __call__(self, by: Optional[list] = None):
        """
        Instaniates Groupby Class which stores a groupby call

        Parameters
        ----------
        by : list
            List of fields to groupby

        """
        self.by = [] if by is None else by
        self.groupby_fields = self._get_groupby_fields()
        self.groupby_call = self._create_groupby_call()
        if self._pre_groupby is not None:
            self.groupby_call += self._pre_groupby
        self.agg = ClusterAgg(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
            groupby_call=self.groupby_call,
            vector_fields=self.vector_fields,
            alias=self.alias,
        )
        return self

    def _get_groupby_fields(self):
        """
        Get what type of groupby field to use
        """
        schema: dict = self.datasets.schema(self.dataset_id)
        self._are_fields_in_schema(self.by, self.dataset_id, schema)
        fields_schema = {k: v for k, v in schema.items() if k in self.by}
        self._check_groupby_value_type(fields_schema)

        return {k: GROUPBY_MAPPING[v] for k, v in fields_schema.items()}

    def _check_groupby_value_type(self, fields_schema: dict):
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

    def mean(self, field: str):
        """
        Convenience method to call avg metric on groupby.

        Parameters
        ----------
        field: str
            The field name to apply the mean aggregation.
        """
        return self.agg({field: "avg"})


class ClusterAgg(APIClient):
    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
        vector_fields: list,
        alias: str,
        groupby_call: Optional[list] = None,
    ):
        self.credentials = credentials
        self.dataset_id = dataset_id
        self.groupby_call = [] if groupby_call is None else groupby_call
        self.vector_fields = vector_fields
        self.alias = alias

    def __call__(  # type: ignore
        self,
        metrics: Optional[dict] = None,
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
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
        self.metrics = {} if metrics is None else metrics
        self._are_fields_in_schema(self.metrics.keys(), self.dataset_id)
        self.metrics_call = self._create_metrics()
        return self.services.cluster.aggregate(
            dataset_id=self.dataset_id,
            alias=self.alias,
            vector_fields=self.vector_fields,
            metrics=self.metrics_call,
            groupby=self.groupby_call,
            page_size=page_size,
            page=page,
            asc=asc,
            flatten=flatten,
        )

    def _create_metrics(self):
        """
        Create metric call
        """
        return [{"name": k, "field": k, "agg": v} for k, v in self.metrics.items()]
