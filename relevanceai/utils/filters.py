from relevanceai._api import APIClient


class Filter(APIClient):
    def __init__(self, field, dataset_id, condition, condition_value):
        super().__init__()

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
                "condition": self.condition,
                "filter_type": "exact_match"
                if self.dtype is "numeric"
                else "word_match",
                "condition": self.condition,
                "condition_value": self.condition_value,
            }
        ]
