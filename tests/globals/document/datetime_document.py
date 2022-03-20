from typing import Dict

from datetime import datetime

from relevanceai.package_utils.make_id import _make_id


def datetime_document() -> Dict:
    document = {
        "sample_1_datetime": datetime.now(),
        "sample_2_datetime": datetime.now(),
    }
    document["_id"] = _make_id(document)
    return document
