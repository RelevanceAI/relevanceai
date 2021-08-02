"""The Transport Class defines a transport as used by the Channel class to communicate with the network.
"""
from json.decoder import JSONDecodeError
from requests import Request
import requests
import time
import traceback
import logging

def create_logger(orig_func, log_file, log_console):
    logger = logging.getLogger(orig_func.__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    if log_file == True:
        file_handler = logging.FileHandler('transport_time.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_console == True:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def logger(orig_func):
    def wrapper(*args, **kwargs):

        if args[0].config.logging is True:

            logger = create_logger(orig_func, args[0].config.log_to_file, args[0].config.log_to_console)

            t1 = time.time()
            results = orig_func(*args, **kwargs)
            t2 = time.time() - t1
            logger.info(f'Ran in {t2} seconds with args {args} and kwargs {kwargs}')

            return results

        else:
            return orig_func(*args, **kwargs)
    
    return wrapper


class Transport:
    """Base class for all VecDB objects
    """
    project: str 
    api_key: str
    
    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}

    
    @logger
    def make_http_request(self, endpoint: str, method: str='GET', parameters: dict={}, output_format: str = "json", 
        base_url: str=None, verbose: bool = True):
        """Make the HTTP request
        Args:
            endpoint: The endpoint from the documentation to use
            method_type: POST or GET request
        """
        if base_url is None:
            base_url = self.base_url
        for i in range(self.config.number_of_retries):
            if verbose: print("URL you are trying to access:" + self.base_url + endpoint) 
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

                else:
                    if verbose: print(response) 
                    if verbose: print(response.content.decode()) 
                    print('Response failed, but re-trying') 
                    continue
            
            except ConnectionError as error:
                # Print the error
                traceback.print_exc()
                print("Connection error but re-trying.") 
                time.sleep(self.config.seconds_between_retries)
                continue

            except JSONDecodeError as error:
                print('No Json available') 
                print(response)

            print('Response failed, stopped trying') 
            return 


