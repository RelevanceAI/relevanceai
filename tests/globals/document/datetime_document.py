from typing import Dict

from datetime import datetime


def datetime_document(_id: str) -> Dict:
    return {
        "_id": _id,
        "sample_1_datetime": datetime.now(),
        "sample_2_datetime": datetime.now(),
    }
