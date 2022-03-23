from relevanceai.constant.config import Config
from relevanceai.constant.links import *

CONFIG_PATH = "relevanceai/constants/config.ini"
CONFIG = Config(CONFIG_PATH)

MAX_CACHESIZE = (
    int(CONFIG["cache.max_size"]) if CONFIG["cache.max_size"] != "None" else None
)

TRANSIT_ENV_VAR = "_IS_ANALYTICS_IN_TRANSIT"

GLOBAL_DATASETS = ["_mock_dataset_"]

DATASETS = [
    "games",
    "ecommerce_1",
    "ecommerce_2",
    "ecommerce_3",
    "online_retail",
    "news",
    "flipkart",
    "realestate2",
]


MB_TO_BYTE = 1024 * 1024
LIST_SIZE_MULTIPLIER = 3

SUCCESS_CODES = [200]
RETRY_CODES = [400, 404]
HALF_CHUNK_CODES = [413, 524]
