from relevanceai._api import APIClient


class Filter(APIClient):
    def __init__(self, field, dataset_id, condition, condition_value, **kwargs):
        super().__init__(**kwargs)

        self.field = field
        self.dataset_id = dataset_id
        self.condition = condition
        self.condition_value = condition_value

    @property
    def dtype(self):
        schema = self.datasets.schema(self.dataset_id)
        return schema[self.field]

    def get(self):
        return [
            {
                "field": self.field,
                "filter_type": "numeric" if self.dtype == "numeric" else "exact_match",
                "condition": self.condition,
                "condition_value": self.condition_value,
            }
        ]
