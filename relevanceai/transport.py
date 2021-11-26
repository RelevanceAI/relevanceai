"""The Transport Class defines a transport as used by the Channel class to communicate with the network.
"""
import time
import traceback
import json
from typing import Union
from relevanceai.config import Config
from json.decoder import JSONDecodeError
from relevanceai.logger import AbstractLogger

import requests
from requests import Request

from relevanceai.errors import APIError


class Transport:
    """Base class for all relevanceai objects"""

    project: str
    api_key: str
    base_url: str
    config: Config
    logger: AbstractLogger

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}

    def make_http_request(
        self,
        endpoint: str,
        method: str = "GET",
        parameters: dict = {},
        base_url: str = None
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
            base_url = self.base_url

        retries = int(self.config.get_option("retries.number_of_retries"))
        seconds_between_retries = int(self.config.get_option("retries.seconds_between_retries"))

        for _ in range(retries):

            self.logger.info(
                "URL you are trying to access:" + base_url + endpoint)
            try:
                req = Request(
                    method=method.upper(),
                    url=base_url + endpoint,
                    headers=self.auth_header,
                    json=parameters if method.upper() == "POST" else {},
                    params=parameters if method.upper() == "GET" else {},
                ).prepare()

                with requests.Session() as s:
                    response = s.send(req)

                if response.status_code == 200:
                    self._log_response_success(base_url, endpoint)
                    self._log_response_time(base_url, endpoint, time.perf_counter() - start_time)
                    return response.json()

                elif response.status_code == 404:
                    self._log_response_fail(base_url, endpoint, response.status_code, response.content.decode())
                    raise APIError(response.content.decode())

                else:
                    self._log_response_fail(base_url, endpoint, response.status_code, response.content.decode())
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
        self.logger.error(f"Response failed ({base_url + endpoint}) (Status: {status_code} Response: {content})")

    def _log_connection_error(self, base_url, endpoint):
        self.logger.error(f"Connection error but re-trying. ({base_url + endpoint})")

    def _log_no_json(self, base_url, endpoint, status_code, content):
        self.logger.error(f"No JSON Available ({base_url + endpoint}) (Status: {status_code} Response: {content})")


