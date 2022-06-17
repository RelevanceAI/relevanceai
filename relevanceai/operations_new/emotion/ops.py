from relevanceai.operations_new.emotion.base import EmotionBase
from relevanceai.operations_new.apibase import OperationAPIBase


class EmotionOps(EmotionBase, OperationAPIBase):
    """
    Add emotions
    """

    @property
    def name(self):
        return "emotion"
