import re

from relevanceai._api import APIClient


class Filter(APIClient):
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

    Exists
    ==============

    Exists filtering can be accessed in a simple way.

    .. code-block::

        old_filters = [
            {
                "field": "brand",
                "filter_type": "exists",
                "condition": "==",
                "condition_value": " "
            }
        ]

        new_filters = dataset["brand"].exists()

        old_filters = [
            {
                "field": "brand",
                "filter_type": "exists",
                "condition": "!=",
                "condition_value": " "
            }
        ]

        new_filters = dataset["brand"].not_exists()

    Contains
    ==============

    Same with contains.

    .. code-block::

        old_filters = [
            {
                "field": "description",
                "filter_type": "contains",
                "condition": "==",
                "condition_value": "Durian BID"
            }
        ]

        new_filters = dataset["description"].contains("Durian BID")

    Dates
    ==============

    Date filtering

    .. code-block::

        old_filters = [
            {
                "field": ""insert_date_"",
                "filter_type": "date",
                "condition": "==",
                "condition_value": "2020-07-01"
            }
        ]

        new_filters = dataset["_insert_date"].date(2020-07-01")

    """

    def __init__(self, field, dataset_id, condition, condition_value, **kwargs):
        super().__init__(kwargs["credentials"])
        kwargs.pop("credentials")

        self.field = field
        self.dataset_id = dataset_id
        self.condition = condition
        self.condition_value = condition_value

        self.date = Filter.is_date(condition_value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def is_date(value: str):
        """
        checks if value is of format

        """
        try:
            value = value.replace("-", "")
            expression = r"^\d{4}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])$"
            if re.findall(expression, value):
                return True
            else:
                return False
        except:
            return False

    @property
    def dtype(self):
        schema = self.datasets.schema(self.dataset_id)
        return schema[self.field]

    def get(self):
        if hasattr(self, "filter_type"):
            filter_type = self.filter_type
        else:
            filter_type = "numeric" if self.dtype == "numeric" else "exact_match"

        if self.date:
            filter_type = "date"

        return [
            {
                "field": self.field,
                "filter_type": filter_type,
                "condition": self.condition,
                "condition_value": self.condition_value,
            }
        ]
