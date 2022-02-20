import analytics
import asyncio
import json
from typing import Callable

from functools import wraps

from relevanceai.config import CONFIG
from relevanceai.json_encoder import json_encoder
from relevanceai.logger import FileLogger


def is_tracking_enabled():
    if CONFIG.is_field("mixpanel.is_tracking_enabled", CONFIG.config):
        return CONFIG.get_field("mixpanel.is_tracking_enabled", CONFIG.config)


def get_json_size(json_obj):
    # Returns it in bytes
    return len(json.dumps(json_obj).encode("utf-8")) / 1024


def track(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(args[0], "_is_analytics_in_transit"):
            if args[0]._is_analytics_in_transit:
                return func(*args, **kwargs)

        args[0]._is_analytics_in_transit = True

        try:
            if is_tracking_enabled():

                async def send_analytics():
                    user_id = args[0].firebase_uid
                    event_name = f"pysdk-{func.__name__}"
                    kwargs.update(dict(zip(func.__code__.co_varnames, args)))
                    properties = {
                        "args": args,
                        "kwargs": kwargs,
                    }
                    if user_id is not None:
                        # TODO: Loop through the properties and remove anything
                        # greater than 5kb
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
        try:
            return func(*args, **kwargs)
        finally:
            args[0]._is_analytics_in_transit = False

    return wrapper


def identify(func: Callable):
    def wrapper(*args, **kwargs):
        try:
            if is_tracking_enabled():
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
