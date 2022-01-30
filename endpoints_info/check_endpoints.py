import os
import requests


ENDPOINT_INFO_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(ENDPOINT_INFO_PATH, ".."))
API_PATH = os.path.join(ROOT_PATH, "relevanceai/api")
MODULES = os.listdir(API_PATH)
MODULES.remove("__pycache__")

ENDPOINTS = list(
    requests.get("https://gateway-api-aueast.relevance.ai/latest/openapi.json")
    .json()["paths"]
    .keys()
)
NOT_NEEDED_ENDPOINTS = [
    "/datasets/infer_schema",
    "/datasets/{dataset_id}/store_encoders_pipeline",
    "/datasets/{dataset_id}/vector_mappings",
    "/services/encoders/numeric_fields",
    "/services/encoders/categories",
    "/services/encoders/dictionary",
    "/services/encoders/encode",
    "/services/encoders/bulk_encode",
    "/deployables/create",
    "/deployables/{deployable_id}/share",
    "/deployables/{deployable_id}/private",
    "/deployables/{deployable_id}/update",
    "/deployables/{deployable_id}/get",
    "/deployables/delete",
    "/deployables/list",
]


endpoint_in_sdk = []
endpoint_not_in_sdk = [
    endpoint for endpoint in ENDPOINTS if endpoint not in NOT_NEEDED_ENDPOINTS
]
for module in MODULES:
    with open(os.path.join(API_PATH, module)) as f:
        lines = f.readlines()
        for line in lines:
            for endpoint in endpoint_not_in_sdk:
                if endpoint in line:
                    endpoint_in_sdk.append(endpoint)
                    endpoint_not_in_sdk.remove(endpoint)


with open(os.path.join(ENDPOINT_INFO_PATH, "endpoint_in_sdk.txt"), "w") as f:
    for endpoint in endpoint_in_sdk:
        f.write(f"{endpoint}\n")

with open(os.path.join(ENDPOINT_INFO_PATH, "endpoint_not_in_sdk.txt"), "w") as f:
    for endpoint in endpoint_not_in_sdk:
        f.write(f"{endpoint}\n")
