"""The Transport Class defines a transport as used by the Channel class to communicate with the network.
"""
import requests
import time

class Transport:
    """Base class for all VecDB objects
    """
    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}
    
    def make_http_request(self, endpoint: str, method: str='GET', parameters: dict={}):
        """Make the HTTP request
        Args:
            endpoint: The endpoint from the documentation to use
            method_type: POST or GET request
        """
        for i in range(self.config.number_of_retries):
            try:
                if method == 'POST':
                    response = requests.post(
                        url=self.base_url + endpoint,
                        headers=self.auth_header,
                        json=parameters
                    )
                elif method == "GET":
                    response = requests.get(
                        url=self.base_url + endpoint,
                        headers=self.auth_header,
                        params=parameters
                    )
                else:
                    raise ValueError(f"You require a GET or a POST method, not {method}.")
                if response.status_code == 200:
                    return response.json()
                else:
                    try:
                        print("URL you are trying to access:" + self.base_url + endpoint)
                        return response.json()
                    except JSONDecodeError:
                        return response.content.decode()
            except ConnectionError as error:
                # Print the error
                import traceback
                traceback.print_exc()
                print("Connection error but re-trying.")
                time.sleep(self.config.seconds_between_retries)
                continue


