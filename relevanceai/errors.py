"""Missing field error
"""
class RelevanceAIError(Exception):
    """base class for all errors
    """

class MissingFieldError(RelevanceAIError):
    """Error handling for missing fields"""


class APIError(RelevanceAIError):
    """Error related to API"""

class ClusteringResultsAlreadyExistsError(RelevanceAIError):
    """Error is raised when the clustering dataset already exists
    """
    pass
