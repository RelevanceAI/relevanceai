from typing import Dict

from datetime import datetime

from relevanceai.dataset.crud.helpers import make_id


def datetime_document(_id: str) -> Dict:
    document = {
        "sample_1_datetime": datetime.now(),
        "sample_2_datetime": datetime.now(),
    }
    document["_id"] = make_id(document)
    return document
