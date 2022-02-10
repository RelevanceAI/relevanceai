import analytics

from typing import Callable

from relevanceai.config import CONFIG


def enable_tracking():
    return CONFIG.get_field("mixpanel.enable_tracking", CONFIG.config)


def track(func: Callable):
    def wrapper(*args, **kwargs):
        try:
            if enable_tracking():
                user_id = args[0].firebase_uid
                event = f"pysdk-{func.__name__}"
                kwargs.update(dict(zip(func.__code__.co_varnames, args)))
                self = kwargs["self"]
                kwargs.pop("self")
                properties = {
                    "args": args[1:],
                    "kwargs": kwargs,
                }
                analytics.track(user_id=user_id, event=event, properties=properties)
                kwargs["self"] = self
        except Exception as e:
            print(e)

        return func(**kwargs)

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
                analytics.identify(user_id, traits)
        except Exception as e:
            print(e)

        return func(*args, **kwargs)

    return wrapper
