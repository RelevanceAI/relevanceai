import sys
import uuid


def make_id(document):
    _id = str(uuid.uuid3(uuid.NAMESPACE_DNS, str(document)))
    return _id


def str2bool(v: str):
    return v.lower() in ("yes", "true", "t", "1")


def getsizeof(obj, seen=None):
    """Recursively finds size of objects"""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([getsizeof(v, seen) for v in obj.values()])
        size += sum([getsizeof(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += getsizeof(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([getsizeof(i, seen) for i in obj])

    return size
