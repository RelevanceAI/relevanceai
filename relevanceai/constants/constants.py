from relevanceai.constants.config import Config


CONFIG_PATH = "relevanceai/constants/config.ini"
CONFIG = Config(CONFIG_PATH)

MAX_CACHESIZE = (
    int(CONFIG["cache.max_size"]) if CONFIG["cache.max_size"] != "None" else None
)

TRANSIT_ENV_VAR = "_IS_ANALYTICS_IN_TRANSIT"

GLOBAL_DATASETS = ["_mock_dataset_"]
