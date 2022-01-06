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


class JSONEncoderUtils:
    def json_encoder(self, obj):
        """
        Converts object so it is json serializable
        If you want to add your own mapping,
        customize it this way;

        Example
        --------

        YOu can use our JSON encoder easily.
        >>> docs = [{"value": np.nan}]
        >>> client.json_encoder(docs)

        If you want to use FastAPI's json encoder, do this:
        >>> from fastapi import jsonable_encoder
        >>> client.json_encoder = jsonable_encoder

        """
        # Loop through iterators and convert
        if isinstance(
            obj, (list, set, frozenset, GeneratorType, tuple, collections.deque)
        ):
            encoded_list = []
            for item in obj:
                encoded_list.append(self.json_encoder(item))
            return encoded_list

        # Loop through dictionaries and convert
        if isinstance(obj, dict):
            encoded_dict = {}
            for key, value in obj.items():
                encoded_key = self.json_encoder(key)
                encoded_value = self.json_encoder(value)
                encoded_dict[encoded_key] = encoded_value
            return encoded_dict

        # Custom conversions
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, (np.ndarray, np.generic)):
            return self.json_encoder(obj.tolist())
        if isinstance(obj, pd.DataFrame):
            return self.json_encoder(obj.to_dict())
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
            return ENCODERS_BY_TYPE[type(obj)](obj)

        raise ValueError(f"{obj} ({type(obj)}) cannot be converted to JSON format")
