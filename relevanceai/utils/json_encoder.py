"""Json Encoder utility

To invoke JSON encoder:

```
    from relevanceai import json_encoder
```

"""
import datetime
import dataclasses
from enum import Enum
from pathlib import PurePath
from types import GeneratorType
import numpy as np
import collections
import pandas as pd

from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
)

from enum import Enum
from types import GeneratorType
import datetime
from uuid import UUID
from collections import deque
from pathlib import Path
from typing import Any

# Taken from pydanitc.json
ENCODERS_BY_TYPE = {
    bytes: lambda o: o.decode(),
    datetime.date: lambda o: o.isoformat(),
    datetime.datetime: lambda o: o.isoformat(),
    datetime.time: lambda o: o.isoformat(),
    datetime.timedelta: lambda td: td.total_seconds(),
    Enum: lambda o: o.value,
    frozenset: list,
    deque: list,
    GeneratorType: list,
    IPv4Address: str,
    IPv4Interface: str,
    IPv4Network: str,
    IPv6Address: str,
    IPv6Interface: str,
    IPv6Network: str,
    Path: str,
    set: list,
    UUID: str,
}


def json_encoder(obj: Any, force_string: bool = False):
    """Converts object so it is json serializable
    If you want to add your own mapping,
    customize it this way;

    Parameters
    ------------
    obj: Any
        The object to convert
    force_string: bool
        If True, forces the object to a string representation. Used mainly for
        analytics tracking.

    Example
    --------

    YOu can use our JSON encoder easily.
    >>> documents = [{"value": np.nan}]
    >>> client.json_encoder(documents)

    If you want to use FastAPI's json encoder, do this:
    >>> from fastapi import jsonable_encoder
    >>> client.json_encoder = jsonable_encoder

    """
    # Loop through iterators and convert
    if isinstance(obj, (list, set, frozenset, GeneratorType, tuple, collections.deque)):
        encoded_list = []
        for item in obj:
            encoded_list.append(json_encoder(item, force_string=force_string))
        return encoded_list

    # Loop through dictionaries and convert
    if isinstance(obj, dict):
        encoded_dict = {}
        for key, value in obj.items():
            encoded_key = json_encoder(key, force_string=force_string)
            encoded_value = json_encoder(value, force_string=force_string)
            encoded_dict[encoded_key] = encoded_value
        return encoded_dict

    # Custom conversions
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, (np.ndarray, np.generic)):
        return json_encoder(obj.tolist(), force_string=force_string)
    if isinstance(obj, pd.DataFrame):
        return json_encoder(obj.to_dict(), force_string=force_string)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (str, int, type(None))):
        return obj
    if isinstance(obj, float):
        if pd.isna(obj):
            return None
        else:
            return obj

    if type(obj) in ENCODERS_BY_TYPE:
        return ENCODERS_BY_TYPE[type(obj)](obj)  # type: ignore

    if force_string:
        return repr(obj)

    raise ValueError(f"{obj} ({type(obj)}) cannot be converted to JSON format")


class JSONEncoderUtils:
    def json_encoder(self, obj, force_string: bool = False):
        return json_encoder(obj, force_string=force_string)
