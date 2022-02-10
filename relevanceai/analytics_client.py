import analytics

from typing import Callable


def track(func: Callable):
    def wrapper(*args, **kwargs):
        user_id = args[0].firebase_uid
        event = func.__name__

        kwargs.update(dict(zip(func.__code__.co_varnames, args)))
        self = kwargs["self"]
        kwargs.pop("self")
        properties = {
            "args": args[1:],
            "kwargs": kwargs,
        }
        analytics.track(user_id=user_id, event=event, properties=properties)

        kwargs["self"] = self
        return func(**kwargs)

    return wrapper


def identify(func: Callable):
    def wrapper(*args, **kwargs):
        user_id = args[0].firebase_uid
        region = args[0].region
        traits = {
            "region": region,
        }
        analytics.identify(user_id, traits)
        return func(*args, **kwargs)

    return wrapper
