"""
All operations related to sentiment
"""
from typing import List
from relevanceai.operations_new.sentiment.transform import SentimentTransform
from relevanceai.operations_new.ops_base import OperationAPIBase


class SentimentOps(SentimentTransform, OperationAPIBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_filters(text_fields: List[str], model_name: str) -> List[str]:

        output_fields = [
            f"_sentiment_.{text_field}.{model_name}.sentiment"
            for text_field in text_fields
        ]

        if len(text_fields) > 1:
            iters = len(text_fields) ** 2

            filters: list = []
            for i in range(iters):
                binary_array = [character for character in str(bin(i))][2:]
                mixed_mask = ["0"] * (
                    len(text_fields) - len(binary_array)
                ) + binary_array
                mask = [int(value) for value in mixed_mask]
                # Creates a binary mask the length of fields provided
                # for two fields, we need 4 iters, going over [(0, 0), (1, 0), (0, 1), (1, 1)]

                condition_value = [
                    {
                        "field": field if mask[index] else output_field,
                        "filter_type": "exists",
                        "condition": "==" if mask[index] else "!=",
                        "condition_value": "",
                    }
                    for index, (field, output_field) in enumerate(
                        zip(text_fields, output_fields)
                    )
                ]
                filters += [{"filter_type": "or", "condition_value": condition_value}]

        else:  # Special Case when only 1 field is provided
            condition_value = [
                {
                    "field": text_fields[0],
                    "filter_type": "exists",
                    "condition": "==",
                    "condition_value": " ",
                },
                {
                    "field": f"_sentiment_.{text_fields[0]}.{model_name}.sentiment",
                    "filter_type": "exists",
                    "condition": "!=",
                    "condition_value": " ",
                },
            ]
            filters = condition_value

        return filters
