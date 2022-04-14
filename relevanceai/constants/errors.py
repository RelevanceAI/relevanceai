from typing import Optional

MISSING_FIELD = "Missing field in dataset"

API_ERROR = "Unfortunately you have encountered an API Error! If this is urgent, please submit a ticket to support@relevance.ai"

CLUSTERING_ALREADY_EXISTS = "Clustering already exists for this Field"

NO_DOCUMENTS = "No Documents were returned"

NO_MODEL = "No Model was specified"


class RelevanceAIError(Exception):
    """_Base class for all errors"""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


class FieldNotFoundError(RelevanceAIError):
    """Error handling for missing fields"""

    def __init__(self, *args, **kwargs):
        message = MISSING_FIELD
        super().__init__(message)


class APIError(RelevanceAIError):
    """Error related to API"""

    def __init__(self, *args, **kwargs):
        if args:
            message = args[0]
        else:
            message = API_ERROR
        super().__init__(message)


class ClusteringResultsAlreadyExistsError(RelevanceAIError):
    """Error for when clustering results already exist"""

    def __init__(self, *args, **kwargs):
        message = CLUSTERING_ALREADY_EXISTS
        super().__init__(message)


class NoDocumentsError(RelevanceAIError):
    """Error for when no documents are retrieved for an operation."""

    def __init__(self, *args, **kwargs):
        message = NO_DOCUMENTS
        super().__init__(message)


class NoModelError(RelevanceAIError):
    """Error for when no model is specified when clustering."""

    def __init__(self, **kwargs):
        message = NO_MODEL
        super().__init__(message)


class TokenNotFoundError(RelevanceAIError):
    """"""

    def __init__(self, *args, **kwargs):
        message = ""
        super().__init__(message)


class ProjectNotFoundError(RelevanceAIError):
    """"""

    def __init__(self, *args, **kwargs):
        message = ""
        super().__init__(message)


class APIKeyNotFoundError(RelevanceAIError):
    """"""

    def __init__(self, *args, **kwargs):
        message = ""
        super().__init__(message)


class FireBaseUIDNotFoundError(RelevanceAIError):
    """"""

    def __init__(self, *args, **kwargs):
        message = ""
        super().__init__(message)


class RegionNotFoundError(RelevanceAIError):
    """"""

    def __init__(self, *args, **kwargs):
        message = ""
        super().__init__(message)


class SetArgumentError(RelevanceAIError):
    def __init__(self, argument, *args, **kwargs):
        message = f"You are missing a {argument}. Please set using the argument {argument}='...'."
        super().__init__(message)


class MissingClusterError(RelevanceAIError):
    """Error for missing clusters"""

    def __init__(self, alias, *args, **kwargs):
        message = f"No clusters with alias `{alias}`. Please check the schema."
        super().__init__(message)


class MissingPackageError(RelevanceAIError):
    def __init__(self, package, version: Optional[str] = None, *args, **kwargs):
        message = f"You need to install {package}! `pip install {package}`."
        if version is None:
            message = message
        else:
            message = message.replace("{package}", f"{package}=={version}")
        super().__init__(message)


class MissingPackageExtraError(RelevanceAIError):
    def __init__(self, extra, version: Optional[str] = None, *args, **kwargs):
        message = f"You need to install the package extra [{extra}]! `pip install RelevanceAI[{extra}]`."

        if version is None:
            message = message
        else:
            message = message.replace(
                f"RelevanceAI[{extra}]", f"RelevanceAI[{extra}=={version}]"
            )
        super().__init__(message)


class ModelNotSupportedError(RelevanceAIError):
    def __init__(self, *args, **kwargs):
        message = "We do not support this kind of model for vectorization"
        super().__init__(message)
