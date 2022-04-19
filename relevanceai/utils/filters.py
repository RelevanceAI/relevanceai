from relevanceai._api import APIClient

"""
Filters have been designed to become more pythonic.

.. code-block::

    old filters = [
        {
            "field": "product_name",
            "filter_type": "exact_match",
            "condition": "==",
            "condition_value": "Durian Leather 2 Seater Sofa"
        }
    ]

    new_filters = dataset["product_name"] == "Durian Leather 2 Seater Sofa"
    # Produces the same as above

    older_filters = [
        {
            "field": "rank",
            "filter_type": "numeric",
            "condition": ">=",
            "condition_value": 2
        },
        {
            "field": "rank",
            "filter_type": "numeric",
            "condition": "<",
            "condition_value": 3
        }
    ]

    new_filters = (dataset["rank"] >= 2) + (dataset["rank"] < 3)

"""


class Filter(APIClient):
    def __init__(self, field, dataset_id, condition, condition_value, **kwargs):
        super().__init__(kwargs["credentials"])
        kwargs.pop("credentials")

        self.field = field
        self.dataset_id = dataset_id
        self.condition = condition
        self.condition_value = condition_value

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def dtype(self):
        schema = self.datasets.schema(self.dataset_id)
        return schema[self.field]

    def get(self):
        if hasattr(self, "filter_type"):
            filter_type = self.filter_type
        else:
            filter_type = "numeric" if self.dtype == "numeric" else "exact_match"
        return [
            {
                "field": self.field,
                "filter_type": filter_type,
                "condition": self.condition,
                "condition_value": self.condition_value,
            }
        ]
