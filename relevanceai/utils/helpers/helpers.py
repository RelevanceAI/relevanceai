import uuid


def make_id(document):
    _id = str(uuid.uuid3(uuid.NAMESPACE_DNS, str(document)))
    return _id


def str2bool(v: str):
    return v.lower() in ("yes", "true", "t", "1")
