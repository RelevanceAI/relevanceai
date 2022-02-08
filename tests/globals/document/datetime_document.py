from typing import Dict

from datetime import datetime


def datetime_document(id: str) -> Dict:
    return {
        "_id": id,
        "sample_1_datetime": datetime.now(),
        "sample_2_datetime": datetime.now(),
    }
