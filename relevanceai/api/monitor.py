"""All Dataset related functions
"""
from relevanceai.base import Base


class Monitor(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

    def health(
        self, dataset_id: str, output_format: str = "json", verbose: bool = True
    ):
        """ 
        Gives you a summary of the health of your vectors, e.g. how many documents with vectors are missing, how many documents with zero vectors 
        
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/monitor/health",
            method="GET",
            output_format=output_format,
            verbose=verbose,
        )


    def stats(self, dataset_id: str, output_format: str = "json", verbose: bool = True):
        """ 
        All operations related to monitoring
        
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/monitor/stats",
            method="GET",
            output_format=output_format,
            verbose=verbose,
        )
