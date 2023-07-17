import os
import json
import atexit
import requests
from fastapi.routing import APIRoute
from relevanceai.auth import config


def routes_to_steps(api_routes, url, id_suffix=""):
    tfs_list = []
    id_list = []
    for route in api_routes:
        if isinstance(route, APIRoute):
            uid = route.unique_id + id_suffix
            id_list.append(uid)
            input_schema = {}
            if route.body_field:
                input_schema = json.loads(route.body_field.type_.schema_json())

            for k, v in input_schema["properties"].items():
                if "frontend" in v:
                    v["metadata"] = v["frontend"]
                    del v["frontend"]
                if "title" in v:
                    if "metadata" not in v:
                        v["metadata"] = {}
                    v["metadata"]["title"] = v["title"]
                    del v["title"]
                if "description" in v:
                    if "metadata" not in v:
                        v["metadata"] = {}
                    v["metadata"]["description"] = v["description"]
                    del v["description"]

            output_schema = {}
            if route.response_field:
                output_schema = json.loads(route.response_field.type_.schema_json())

            if url.endswith("/"):
                url = url[:-1]
            route_path = route.path
            if route_path.startswith("/"):
                route_path = route_path[1:]
            full_path = url + "/" + route_path

            tfs_list.append(
                {
                    "_id": route.unique_id,
                    "transformation_id": uid,
                    "name": route.summary if route.summary else route.name,
                    "description": route.description,
                    "studio_api_path": full_path,
                    "execution_type": "studio-api",
                    "input_schema": input_schema,
                    "output_schema": output_schema,
                }
            )
    return tfs_list, id_list


def list_steps():
    url = f"https://api-{config.auth.region}.stack.tryrelevance.com"
    results = requests.post(
        f"{url}/latest/studios/transformations/custom/list",
        headers=config.auth.headers,
        json={"page": 1, "page_size": 10},
    )
    print("List of transformations: ", results.json())
    print("Trace-id ", results.headers.get("x-trace-id"))


def cleanup_steps(transformation_id_list):
    url = f"https://api-{config.auth.region}.stack.tryrelevance.com"
    results = requests.post(
        f"{url}/latest/studios/transformations/custom/bulk_delete",
        headers=config.auth.headers,
        json={"ids": transformation_id_list},
    )
    print("Successfully deleted transformations from cloud: ", results.json())
    print("Trace-id ", results.headers.get("x-trace-id"))


def upload_steps(tfs):
    url = f"https://api-{config.auth.region}.stack.tryrelevance.com"
    results = requests.post(
        f"{url}/latest/studios/transformations/custom/bulk_update",
        headers=config.auth.headers,
        json={"updates": tfs},
    )
    print("Uploaded transformations: ", results.json())
    print("Trace-id ", results.headers.get("x-trace-id"))


def connect_steps(
    api_routes, url, id_suffix="", cleanup=True, export_json=False
):
    tfs_list, id_list = routes_to_steps(api_routes, url, id_suffix=id_suffix)
    if export_json:
        import json

        with open("transformation_export.json", "w") as outfile:
            json.dump({"export": tfs_list}, outfile)
    else:
        upload_steps(tfs_list)
    if cleanup:
        atexit.register(cleanup_steps, id_list)
    return tfs_list
