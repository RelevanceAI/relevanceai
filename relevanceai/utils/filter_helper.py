from dataclasses import dataclass


def create_filter(
    field: str, filter_type: str, condition: str = "==", condition_value: str = " "
):
    return [
        {
            "field": field,
            "filter_type": filter_type,
            "condition": condition,
            "condition_value": condition_value,
        }
    ]


@dataclass
class Filter:
    field: str
    filter_type: str
    condition: str = "=="
    condition_value: str = " "
