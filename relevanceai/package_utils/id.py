import uuid


def _make_id(document):
    _id = str(uuid.uuid3(uuid.NAMESPACE_DNS, str(document)))
    return _id
