import io
import os
import requests
from typing import List
from relevanceai.auth import config


def _get_content_bytes(content):
    if isinstance(content, str):
        if "http" in content and "/":
            # online image
            content_bytes = io.BytesIO(requests.get(content).content).getvalue()
        else:
            # local filepath
            content_bytes = io.BytesIO(open(content, "rb").read()).getvalue()
    elif isinstance(content, bytes):
        content_bytes = content
    elif isinstance(content, io.BytesIO):
        content_bytes = content.getvalue()
    else:
        raise TypeError("'content' needs to be of type str, bytes or io.BytesIO.")
    return content_bytes


def _get_file_upload_urls(dataset_id: str, files: List[str]):
    response = requests.post(
        url=f"https://api-{config.auth.region}.stack.tryrelevance.com/latest/datasets/{dataset_id}/get_file_upload_urls",
        headers=config.auth.headers,
        json={"files": files},
    )
    try:
        return response.json()
    except:
        print(response.text)
        print(response.headers)
        raise


def _upload_media(presigned_url: str, media_content: bytes):
    if not isinstance(media_content, bytes):
        raise ValueError(
            f"media needs to be in a bytes format. Currently in {type(media_content)}"
        )
    return requests.put(
        presigned_url,
        data=media_content,
    )


def upload(data, dataset_id: str, filename: str = "temp"):
    data_bytes = _get_content_bytes(data)
    presigned_response = _get_file_upload_urls(dataset_id=dataset_id, files=[filename])
    response = _upload_media(
        presigned_url=presigned_response["files"][0]["upload_url"],
        media_content=data_bytes,
    )
    assert response.status_code == 200
    return presigned_response["files"][0]["url"]
