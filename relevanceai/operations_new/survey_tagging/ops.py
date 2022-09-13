"""
Labelling with API-related functions
"""
from relevanceai.operations_new.survey_tagging.transform import SurveyTagTransform
from relevanceai.operations_new.ops_base import OperationAPIBase


class SurveyTagOps(
    OperationAPIBase,
    SurveyTagTransform,
):  # type: ignore
    """
    Label Operations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
