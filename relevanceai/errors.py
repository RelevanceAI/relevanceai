"""Missing field error
"""


class MissingFieldError(Exception):
    """Error handling for missing fields"""


class APIError(Exception):
    """Error related to API"""

class ClusteringResultsAlredyExistsError(Exception):
    """Error is raised when the clustering dataset already exists
    """
    pass
