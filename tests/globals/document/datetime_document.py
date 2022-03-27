from typing import Dict

from datetime import datetime

from relevanceai.utils import make_id


def datetime_document() -> Dict:
    document = {
        "sample_1_datetime": datetime.now(),
        "sample_2_datetime": datetime.now(),
    }
    document["_id"] = make_id(document)
    return document
