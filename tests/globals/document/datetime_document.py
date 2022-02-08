import uuid

from datetime import datetime


def _sample_datetime_document():
    return {
        "_id": uuid.uuid4().__str__(),
        "sample_1_datetime": datetime.now(),
        "sample_2_datetime": datetime.now(),
    }
