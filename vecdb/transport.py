"""The Transport Class defines a transport as used by the Channel class to communicate with the network.
"""
from json.decoder import JSONDecodeError
from requests import Request
import requests
import traceback
import time
from .logging import Profiler
from .errors import APIError

class Transport:
    """Base class for all VecDB objects
    """
    project: str 
    api_key: str
    
    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}

    
    def make_http_request(self, endpoint: str, method: str='GET', parameters: dict={}, output_format: str = "json", 
        base_url: str=None, verbose: bool = True, retries: bool = None):
        """Make the HTTP request
        Args:
            endpoint: The endpoint from the documentation to use
            method_type: POST or GET request
        """
        
        with Profiler(self.config.log, self.config.logging_level, self.config.log_to_file, self.config.log_to_console, locals()) as log:
            if base_url is None:
                base_url = self.base_url

            if retries is None:
                retries = self.config.number_of_retries
            
            for i in range(retries):
                if verbose: print("URL you are trying to access:" + base_url + endpoint) 
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
                        if verbose: print("Response success!") 
                        if output_format == "json":
                            return response.json()
                        else:
                            return response

                    elif response.status_code == 404:
                        if verbose: print(response.content.decode()) 
                        if verbose: print(f'Response failed (status: {response.status_code} Content: {response.content.decode()})') 
                        raise APIError(response.content.decode())

                    else:
                        if verbose: print(response.content.decode()) 
                        if verbose: print(f'Response failed (status: {response.status_code} Content: {response.content.decode()})') 
                        continue
                
                except ConnectionError as error:
                    # Print the error
                    traceback.print_exc()
                    if verbose: print("Connection error but re-trying.") 
                    time.sleep(self.config.seconds_between_retries)
                    continue

                except JSONDecodeError as error:
                    if verbose:  print('No Json available') 
                    print(response)

                if verbose: print('Response failed, stopped trying') 
                raise APIError(response.content.decode())
            return response
