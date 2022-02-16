import analytics
import asyncio
import json
from typing import Callable

from functools import wraps

from relevanceai.config import CONFIG
from relevanceai.json_encoder import json_encoder
from relevanceai.logger import FileLogger


def enable_tracking():
    if CONFIG.is_field("mixpanel.enable_tracking", CONFIG.config):
        return CONFIG.get_field("mixpanel.enable_tracking", CONFIG.config)


def get_json_size(json_obj):
    # Returns it in bytes
    return len(json.dumps(json_obj).encode("utf-8")) / 1024


def track(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            if enable_tracking():

                async def send_analytics():
                    user_id = args[0].firebase_uid
                    event_name = f"pysdk-{func.__name__}"
                    kwargs.update(dict(zip(func.__code__.co_varnames, args)))
                    properties = {
                        "args": args,
                        "kwargs": kwargs,
                    }
                    if user_id is not None:
                        # Upsert/inserts/updates are too big to track
                        if (
                            "insert" in event_name
                            or "upsert" in event_name
                            or "update" in event_name
                            or "fit" in event_name
                            or "predict" in event_name
                            or get_json_size(
                                json_encoder(properties, force_string=True)
                            )
                            > 30
                        ):
                            analytics.track(
                                user_id=user_id,
                                event=event_name,
                            )
                        else:
                            analytics.track(
                                user_id=user_id,
                                event=event_name,
                                properties=json_encoder(properties, force_string=True),
                            )

                asyncio.ensure_future(send_analytics())
        except Exception as e:
            pass

        return func(*args, **kwargs)

    return wrapper


def identify(func: Callable):
    def wrapper(*args, **kwargs):
        try:
            if enable_tracking():
                user_id = args[0].firebase_uid
                region = args[0].region
                traits = {
                    "region": region,
                }
                if user_id is not None:
                    analytics.identify(user_id, traits)
        except Exception as e:
            pass

        return func(*args, **kwargs)

    return wrapper
