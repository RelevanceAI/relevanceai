from relevanceai.constants.config import CONFIG

MAX_CACHESIZE = (
    int(CONFIG["cache.max_size"]) if CONFIG["cache.max_size"] != "None" else None
)
