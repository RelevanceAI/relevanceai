import analytics
import asyncio
import json
import os
import copy

from typing import Callable
from base64 import b64decode as decode
from functools import wraps

from relevanceai.constants import CONFIG
from relevanceai.constants import TRANSIT_ENV_VAR

from relevanceai.utils.json_encoder import json_encoder
from relevanceai.utils.decorators.thread import fire_and_forget


def is_tracking_enabled():
    if CONFIG.is_field("mixpanel.is_tracking_enabled", CONFIG.config):
        return CONFIG.get_field("mixpanel.is_tracking_enabled", CONFIG.config)


def get_json_size(json_obj):
    # Returns it in bytes
    return len(json.dumps(json_obj).encode("utf-8")) / 1024


def track(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv(TRANSIT_ENV_VAR) == "TRUE":
            # This env variable is used to track whether to send to MixPanel or not. If True, it does not and immediately
            # goes to function. Otherwise, it logs to MixPanel.
            return func(*args, **kwargs)

        os.environ[TRANSIT_ENV_VAR] = "TRUE"

        try:
            if is_tracking_enabled():

                @fire_and_forget
                def send_analytics():
                    properties = {}
                    if "firebase_uid" in kwargs:
                        user_id = kwargs["firebase_uid"]
                    elif hasattr(args[0], "firebase_uid"):
                        user_id = args[0].firebase_uid
                    else:
                        user_id = "firebase_uid_not_detected"

                    event_name = f"pysdk-{func.__name__}"

                    # kwargs.update(dict(zip(func.__code__.co_varnames, args)))
                    # all_kwargs = copy.deepcopy(kwargs)
                    # all_kwargs = all_kwargs.update(
                    #     dict(zip(func.__code__.co_varnames, args))
                    # )

                    # if hasattr(func, "__code__"):
                    additional_args = dict(zip(func.__code__.co_varnames, args))

                    if "dataset_id" in kwargs:
                        properties["dataset_id"] = kwargs["dataset_id"]

                    elif "dataset_id" in additional_args:
                        properties["dataset_id"] = additional_args["dataset_id"]

                    elif hasattr(args[0], "dataset_id"):
                        properties["dataset_id"] = args[0].dataset_id

                    elif "self" in additional_args:
                        if hasattr(additional_args["self"], "dataset_id"):
                            properties["dataset_id"] = additional_args[
                                "self"
                            ].dataset_id

                    full_properties = copy.deepcopy(properties)
                    full_properties.update(
                        {
                            "additional_args": additional_args,
                            "args": args,
                            "kwargs": kwargs,
                        }
                    )
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
                                json_encoder(full_properties, force_string=True)
                            )
                            > 30
                        ):
                            response = analytics.track(
                                user_id=user_id,
                                event=event_name,
                                properties=json_encoder(properties, force_string=True),
                            )
                        else:
                            response = analytics.track(
                                user_id=user_id,
                                event=event_name,
                                properties=json_encoder(
                                    full_properties, force_string=True
                                ),
                            )

                send_analytics()
                # asyncio.ensure_future(send_analytics())
        except Exception as e:
            pass
        try:
            return func(*args, **kwargs)
        finally:
            os.environ[TRANSIT_ENV_VAR] = "FALSE"

    return wrapper


def track_event_usage(event_name: str):
    EVENT_NAME = event_name
    write_key = CONFIG.get_field("mixpanel.write_key", CONFIG.config)
    analytics.write_key = decode(write_key).decode("utf-8")

    def track(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.getenv(TRANSIT_ENV_VAR) == "TRUE":
                return func(*args, **kwargs)

            os.environ[TRANSIT_ENV_VAR] = "TRUE"

            try:
                if is_tracking_enabled():

                    async def send_analytics():
                        user_id = "OPEN_SOURCE_USER"
                        event_name = f"pysdk-{EVENT_NAME}"

                        if user_id is not None:
                            # TODO: Loop through the properties and remove anything
                            # greater than 5kb
                            response = analytics.track(
                                user_id=user_id,
                                event=event_name,
                            )

                    asyncio.ensure_future(send_analytics())
            except Exception as e:
                pass
            try:
                return func(*args, **kwargs)
            finally:
                os.environ[TRANSIT_ENV_VAR] = "FALSE"

        return wrapper

    return track


def identify(func: Callable):
    def wrapper(*args, **kwargs):
        if os.getenv(TRANSIT_ENV_VAR) == "TRUE":
            return func(*args, **kwargs)

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
