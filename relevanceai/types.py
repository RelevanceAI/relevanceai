from typing import Union, Dict, List, Any

JSONValue = Union[str, float, bool, int, None]
JSONDict = Dict[str, Any]
JSONList = List[Any]
JSONObject = Union[JSONValue, JSONDict, JSONList]
