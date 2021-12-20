"""The Transport Class defines a transport as used by the Channel class to communicate with the network.
"""
import time
import traceback
import json
from typing import Union
from json.decoder import JSONDecodeError

from urllib.parse import urlparse

import requests
from requests import Request

from relevanceai.config import Config
from relevanceai.logger import AbstractLogger
from relevanceai.dashboard_mappings import DASHBOARD_MAPPINGS
from relevanceai.errors import APIError


class Transport:
    """Base class for all relevanceai objects"""

    project: str
    api_key: str
    config: Config
    logger: AbstractLogger

    @property
    def _dashboard_request_url(self):
        return self.config.get_option("dashboard.dashboard_request_url")[1:-1]

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}

    @property
    def _search_dashboard_url(self):
        return (
            self.config["dashboard.base_dashboard_url"][1:-1]
            + self.config["dashboard.search_dashboard_endpoint"][1:-1]
        )

    @staticmethod
    def _is_search_in_path(url: str):
        if url is None:
            return False
        result = urlparse(url)
        return "search" in result.path.split("/")

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
        url = self.config.get_option("api.base_url")[:-2]
        version = self.config.get_option("api.base_url")[-2:]
        request_body = {
            dashboard_type: {
                "body": parameters,
                "url": url,
                "version": version,
                "endpoint": endpoint[1:],
                "metadata": parameters,
                "query": parameters.get("query"),
            },
        }
        req = Request(
            method=method.upper(),
            url=self._dashboard_request_url,
            headers=self.auth_header,
            json=request_body,
            # params=parameters if method.upper() == "GET" else {},
        ).prepare()
        with requests.Session() as s:
            response = s.send(req)

        if verbose:
            dashboard_url = (
                self.config["dashboard.base_dashboard_url"][1:-1]
                + DASHBOARD_MAPPINGS[dashboard_type]
            )
            self.print_dashboard_url(dashboard_url)
        return response

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
        return self._log_to_dashboard(
            method=method,
            parameters=parameters,
            endpoint=endpoint,
            dashboard_type="multivector_search",
        )

    def print_dashboard_url(self, dashboard_url):
        print(f"You can now visit the dashboard at {dashboard_url}")

    def make_http_request(
        self,
        endpoint: str,
        method: str = "GET",
        parameters: dict = {},
        base_url: str = None,
        output_format=None,
    ):
        """
        Make the HTTP request
        Parameters
        ----------
        endpoint: string
            The endpoint from the documentation to use
        method_type: string
            POST or GET request
        """
        self._last_used_endpoint = endpoint
        start_time = time.perf_counter()

        if base_url is None:
            # if Transport.is_search_in_path(base_url) and not hasattr(self, "output_format"):
            #     base_url = self.config.get_option("dashboard.base_dashboard_url")[1:-1]
            # else:
            base_url = self.config.get_option("api.base_url")

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

                req = Request(
                    method=method.upper(),
                    url=request_url,
                    headers=self.auth_header,
                    json=parameters if method.upper() == "POST" else {},
                    params=parameters if method.upper() == "GET" else {},
                ).prepare()

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
                elif response.status_code == 404:
                    self._log_response_fail(
                        base_url,
                        endpoint,
                        response.status_code,
                        response.content.decode(),
                    )
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
