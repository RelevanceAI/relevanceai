import os

from relevanceai.constants.config import Config
from relevanceai.constants.links import *

CONFIG_PATH = os.path.dirname(os.path.abspath(__file__)) + "/config.ini"
CONFIG = Config(CONFIG_PATH)

MAX_CACHESIZE = (
    int(CONFIG["cache.max_size"]) if CONFIG["cache.max_size"] != "None" else None
)

TRANSIT_ENV_VAR = "_IS_ANALYTICS_IN_TRANSIT"

GLOBAL_DATASETS = ["_mock_dataset_"]
DATASETS = [
    "coco",
    "games",
    "ecommerce_1",
    "ecommerce_2",
    "ecommerce_3",
    "online_retail",
    "news",
    "flipkart",
    "realestate2",
    "toy_image_caption_coco_image_encoded",
]

MB_TO_BYTE = 1024 * 1024
LIST_SIZE_MULTIPLIER = 3

SUCCESS_CODES = [200]
RETRY_CODES = [400, 404]
HALF_CHUNK_CODES = [413, 524]

US_EAST_1 = "us-east-1"
AP_SOUTEAST_1 = "ap-southeast-1"
OLD_AUSTRALIA_EAST = "old-australia-east"

IMG_EXTS = [
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".png",
    ".gif",
    ".bmp",
    ".eps",
]
