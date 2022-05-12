"""The Transport Class defines a transport as used by the Channel class to communicate with the network.
"""
import os
import asyncio
import codecs
import time
import json
import traceback

from pprint import pprint
from json.decoder import JSONDecodeError
from typing import Optional

from urllib.parse import urlparse

import aiohttp
import requests

from requests import Request

from relevanceai.constants.config import Config
from relevanceai.utils.logger import AbstractLogger, FileLogger
from relevanceai.dashboard.dashboard_mappings import DASHBOARD_MAPPINGS
from relevanceai.constants.errors import APIError
from relevanceai.utils.json_encoder import JSONEncoderUtils
from relevanceai.utils.config_mixin import ConfigMixin

DO_NOT_REPEAT_STATUS_CODES = {400, 401, 413, 404, 422}
_HAS_PRINTED = False


class Transport(JSONEncoderUtils, ConfigMixin):
    """_Base class for all relevanceai objects"""

    project: str
    api_key: str
    config: Config
    logger: AbstractLogger
    request_logger: FileLogger

    def __init__(self, request_log_filename="request.jsonl", **kwargs):

        if os.getenv("DEBUG_REQUESTS") == "TRUE":
            try:
                from appdirs import user_cache_dir
            except:
                raise ModuleNotFoundError("please instal appdirs `pip install appdirs`")

            from relevanceai import __version__

            dir = user_cache_dir("relevanceai", version=__version__)
            os.makedirs(dir, exist_ok=True)

            self.request_logging_fpath = os.path.join(
                dir, request_log_filename
            ).replace("\\", "/")

            from relevanceai.utils import FileLogger

            self.request_logger = FileLogger(fn=self.request_logging_fpath)
            self.hooks = {"response": self.log}

        else:
            self.hooks = None

    def log(self, response, *args, **kwargs):
        """It takes the response from the request and logs the url, path_url, method, status_code, headers,
        content, time, and elapsed time

        Parameters
        ----------
        response
            The response object

        """
        with self.request_logger:
            log = {}
            log["url"] = response.url
            log["path_url"] = response.request.path_url
            log["method"] = response.request.method
            log["headers"] = response.headers
            log["time"] = time.time()
            log["elapsed"] = response.elapsed.microseconds
            try:
                log["body"] = json.loads(response.request.body)
            except:
                log["body"] = {}

            try:
                content = json.loads(response.content)
            except:
                content = response.content

            response = {"send": log, "recv": content}
            pprint(response, sort_dicts=False)

            print()
            print()

    @property
    def _dashboard_request_url(self):
        return self.config.get_option("dashboard.dashboard_request_url")

    @property
    def auth_header(self):
        if not hasattr(self, "_auth_header"):
            return {"Authorization": self.project + ":" + self.api_key}
        return self._auth_header

    @auth_header.setter
    def auth_header(self, value):
        self._auth_header = value

    @property
    def _search_dashboard_url(self):
        return (
            self.config["dashboard.base_dashboard_url"]
            + self.config["dashboard.search_dashboard_endpoint"]
        )

    @staticmethod
    def _is_search_in_path(url: str):
        if url is None:
            return False
        result = urlparse(url)
        split_path = result.path.split("/")
        return "search" in split_path and "services" in split_path

    @property
    def DASHBOARD_TYPES(self):
        return list(DASHBOARD_MAPPINGS.keys())

    def _log_to_dashboard(
        self,
        method: str,
        parameters: dict,
        endpoint: str,
        dashboard_type: str,
        verbose: bool = True,
    ):
        """Log search to dashboard"""
        # Needs to be a supported dashboard type
        if dashboard_type not in self.DASHBOARD_TYPES:
            return
        # Get the URL but not the version
        url = "/".join(self.base_url.split("/")[:-1]) + "/"  # type: ignore
        # Split off the version separately
        version = self.base_url.split("/")[-1]  # type: ignore
        # Parse the endpoint so it becomes 'endpoint/schema' instead of '/endpoint/schema'
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        request_body = {
            dashboard_type: {
                "body": parameters,
                "url": url,
                "version": version,
                "endpoint": endpoint,
                "metadata": parameters,
                "query": parameters.get("query"),
            },
        }
        self.logger.debug(request_body)

        async def run_request():
            req = Request(
                method=method.upper(),
                url=self._dashboard_request_url,
                headers=self.auth_header,
                json=request_body,
                # params=parameters if method.upper() == "GET" else {},
            ).prepare()
            with requests.Session() as s:
                response = s.send(req)

        asyncio.ensure_future(run_request())

        if verbose:
            dashboard_url = (
                self.config["dashboard.base_dashboard_url"]
                + DASHBOARD_MAPPINGS[dashboard_type]
            )
            self.print_dashboard_url(dashboard_url)

    def _link_to_dataset_dashboard(self, dataset_id: str, suburl: str = None):
        """Link to a monitoring dashboard
        Suburl must be one of
        - "monitor"
        - "lookups"
        - "monitor/schema"
        """
        MESSAGE = "You can view your dashboard at: "
        if suburl is None:
            print(
                MESSAGE
                + f"https://cloud.relevance.ai/dataset/{dataset_id}/dashboard/monitor/"
            )
        else:
            print(
                MESSAGE
                + f"https://cloud.relevance.ai/dataset/{dataset_id}/dashboard/{suburl}"
            )

    def _log_search_to_dashboard(self, method: str, parameters: dict, endpoint: str):
        """Log search to dashboard"""
        self._log_to_dashboard(
            method=method,
            parameters=parameters,
            endpoint=endpoint,
            dashboard_type="multivector_search",
        )

    def print_dashboard_message(self, message: str):
        if self.config["dashboard.show_dashboard_link"]:
            print(message)

    def print_dashboard_url(self, dashboard_url):
        if self.config["dashboard.show_dashboard_link"]:
            print(f"You can now visit the dashboard at {dashboard_url}")

    def make_http_request(
        self,
        endpoint: str,
        method: str = "GET",
        parameters: Optional[dict] = None,
        base_url: str = None,
        output_format=None,
        raise_error: bool = True,
    ):
        """
        Make the HTTP request
        Parameters
        ----------
        endpoint: string
            The endpoint from the documentation to use
        method_type: string
            POST or GET request
        raise_error: bool
            If True, you will raise error. This is useful for endpoints that don't
            necessarily need to error.
        """
        parameters = {} if parameters is None else parameters
        self._last_used_endpoint = endpoint
        start_time = time.perf_counter()

        if base_url is None:
            # if Transport.is_search_in_path(base_url) and not hasattr(self, "output_format"):
            #     base_url = self.config.get_option("dashboard.base_dashboard_url")[1:-1]
            # else:
            if hasattr(self, "base_url"):
                base_url = self.base_url  # type: ignore
            else:
                base_url = "https://api.us-east-1.relevance.ai/latest"

        if output_format is None:
            output_format = self.config.get_option("api.output_format")

        retries = int(self.config.get_option("retries.number_of_retries"))
        seconds_between_retries = int(
            self.config.get_option("retries.seconds_between_retries")
        )
        request_url = base_url + endpoint
        for _ in range(retries):

            self.logger.info("URL you are trying to access:" + request_url)
            try:
                if Transport._is_search_in_path(request_url):
                    self._log_search_to_dashboard(
                        method=method, parameters=parameters, endpoint=endpoint
                    )

                # TODO: Add other endpoints in here too
                if method.upper() in {"POST", "PUT"}:
                    req = Request(
                        method=method.upper(),
                        url=request_url,
                        headers=self.auth_header,
                        json=parameters if method.upper() == "POST" else {},
                        hooks=self.hooks,
                    ).prepare()
                elif method.upper() == "GET":
                    # Get requests do not have JSONs - which will error out
                    # cloudfront
                    req = Request(
                        method=method.upper(),
                        url=request_url,
                        headers=self.auth_header,
                        params=parameters if method.upper() == "GET" else {},
                        hooks=self.hooks,
                    ).prepare()

                # if self.enable_request_logging:
                #     print("URL: ", request_url)
                #     if self.enable_request_logging == "full":
                #         print("HEADERS: ", req.headers)
                #         print("BODY: ", req.body)

                with requests.Session() as s:
                    response = s.send(req)

                # Successful response
                if response.status_code == 200:
                    self._log_response_success(base_url, endpoint)
                    self._log_response_time(
                        base_url, endpoint, time.perf_counter() - start_time
                    )

                    if output_format == "json":
                        return response.json()
                    elif output_format == "content":
                        return response.content
                    elif output_format == "status_code":
                        return response.status_code
                    else:
                        return response

                # Cancel bad URLs
                # Logged status codes
                elif response.status_code in DO_NOT_REPEAT_STATUS_CODES:
                    self._log_response_fail(
                        base_url,
                        endpoint,
                        response.status_code,
                        response.content.decode(),
                    )
                    if raise_error:
                        raise APIError(response.content.decode())

                # Retry other errors
                else:
                    self._log_response_fail(
                        base_url,
                        endpoint,
                        response.status_code,
                        response.content.decode(),
                    )
                    continue

            except (ConnectionError) as error:
                # Print the error
                traceback.print_exc()
                self._log_connection_error(base_url, endpoint)
                time.sleep(seconds_between_retries)
                continue

            except JSONDecodeError as error:
                self._log_no_json(base_url, endpoint, response.status_code, response)
                return response

        return response

    async def make_async_http_request(
        self,
        endpoint: str,
        method: str = "GET",
        parameters: Optional[dict] = None,
        base_url: str = None,
        output_format=None,
        raise_error: bool = True,
    ):
        """
        Make an asynchronous HTTP request

        Parameters
        ----------
        endpoint: str
            The Relevance AI endpoint to access

        method_type: str
            Currently only support GET and POST requests

        raise_error: bool
            If True, error is raised rather than just logged.
        """
        parameters = {} if parameters is None else parameters
        self._last_used_endpoint = endpoint
        start_time = time.perf_counter()

        base_url = self.base_url if base_url is None else base_url  # type: ignore
        output_format = (
            self.config.get_option("api.output_format")
            if output_format is None
            else output_format
        )
        retries = int(self.config.get_option("retries.number_of_retries"))
        seconds_between_retries = int(
            self.config.get_option("retries.seconds_between_retries")
        )

        request_url = base_url + endpoint

        for _ in range(retries):
            self.logger.info(f"URL you are trying to access: {request_url}")
            try:
                if Transport._is_search_in_path(request_url):
                    self._log_search_to_dashboard(
                        method=method, parameters=parameters, endpoint=endpoint
                    )

                async with aiohttp.request(
                    method=method.upper(),
                    url=request_url,
                    headers=self.auth_header,
                    json=parameters if method.upper() == "POST" else {},
                    params=parameters if method.upper() == "GET" else {},
                ) as response:

                    if os.getenv("DEBUG_REQUESTS") == "TRUE":
                        self.log(response)

                    if response.status == 200:
                        self._log_response_success(base_url, endpoint)
                        self._log_response_time(
                            base_url, endpoint, time.perf_counter() - start_time
                        )

                        if output_format.lower() == "json":
                            return await response.json()
                        elif output_format.lower() == "content":
                            decoded_content = codecs.decode(
                                await response.content.read()
                            )
                            return decoded_content
                        elif output_format.lower() == "status_code":
                            return response.status
                        else:
                            return response
                    elif response.status in DO_NOT_REPEAT_STATUS_CODES:
                        # Cancel bad URLs
                        # Logged status codes
                        decoded_content = codecs.decode(await response.content.read())
                        self._log_response_fail(
                            base_url, endpoint, response.status, decoded_content
                        )
                        if raise_error:
                            raise APIError(decoded_content)
                    else:
                        # Retry other errors
                        decoded_content = codecs.decode(await response.content.read())
                        self._log_response_fail(
                            base_url, endpoint, response.status, decoded_content
                        )
            except aiohttp.ClientError as error:
                traceback.print_exc()
                self._log_connection_error(base_url, endpoint)
                time.sleep(seconds_between_retries)
                continue
            except JSONDecodeError as error:
                self._log_no_json(base_url, endpoint, response.status, response)
                return response

        return response

    def _log_response_success(self, base_url, endpoint):
        self.logger.success(f"Response success! ({base_url + endpoint})")

    def _log_response_time(self, base_url, endpoint, time):
        self.logger.debug(f"Request ran in {time} seconds ({base_url + endpoint})")

    def _log_response_fail(self, base_url, endpoint, status_code, content):
        self.logger.error(
            f"Response failed ({base_url + endpoint}) (Status: {status_code} Response: {content})"
        )

    def _log_connection_error(self, base_url, endpoint):
        self.logger.error(f"Connection error but re-trying. ({base_url + endpoint})")

    def _log_no_json(self, base_url, endpoint, status_code, content):
        self.logger.error(
            f"No JSON Available ({base_url + endpoint}) (Status: {status_code} Response: {content})"
        )
