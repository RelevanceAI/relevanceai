"""The Transport Class defines a transport as used by the Channel class to communicate with the network.
"""
import time
import traceback
from json.decoder import JSONDecodeError

import requests
from requests import Request

from .errors import APIError


class Transport:
    """Base class for all relevanceai objects"""

    project: str
    api_key: str

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}

    def make_http_request(
        self,
        endpoint: str,
        method: str = "GET",
        parameters: dict = {},
        output_format: str = "json",
        base_url: str = None,
        verbose: bool = True,
        retries: bool = None,
    ):
        """Make the HTTP request
        Args:
            endpoint: The endpoint from the documentation to use
            method_type: POST or GET request
        """

        t1 = time.time()
        if base_url is None:
            base_url = self.base_url

        if retries is None:
            retries = int(self.config.get_option("retries.number_of_retries"))

        for _ in range(retries):
            if verbose:
                self.logger.info("URL you are trying to access:" + base_url + endpoint)
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
                    if verbose:
                        self.logger.success(
                            f"Response success! ({base_url + endpoint})"
                        )
                    time_diff = time.time() - t1
                    self.logger.debug(
                        f"Request ran in {time_diff} seconds ({base_url + endpoint})"
                    )

                    if output_format == "json":
                        return response.json()
                    else:
                        return response

                elif response.status_code == 404:
                    if verbose:
                        self.logger.error(response.content.decode())
                    if verbose:
                        self.logger.error(
                            f"Response failed ({base_url + endpoint}) (status: {response.status_code} Content: {response.content.decode()})"
                        )
                    raise APIError(response.content.decode())

                else:
                    if verbose:
                        self.logger.error(response.content.decode())
                    if verbose:
                        self.logger.error(
                            f"Response failed ({base_url + endpoint}) (status: {response.status_code} Content: {response.content.decode()})"
                        )
                    continue

            except (ConnectionError) as error:
                # Print the error
                traceback.print_exc()
                if verbose:
                    self.logger.error(
                        f"Connection error but re-trying. ({base_url + endpoint})"
                    )
                time.sleep(
                    int(self.config.get_option("retries.seconds_between_retries"))
                )
                continue

            except JSONDecodeError as error:
                if verbose:
                    self.logger.error(f"No Json available ({base_url + endpoint})")
                self.logger.error(response)

            if verbose:
                self.logger.error(
                    f"Response failed, stopped trying ({base_url + endpoint})"
                )
            raise APIError(response.content.decode())

        return response
