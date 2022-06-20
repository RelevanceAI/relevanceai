from relevanceai.operations_new.emotion.base import EmotionBase
from relevanceai.operations_new.apibase import OperationAPIBase


class EmotionOps(OperationAPIBase, EmotionBase):
    """
    Add emotions
    """

    def __init__(
        self,
        text_fields: list,
        model_name="joeddav/distilbert-base-uncased-go-emotions-student",
        output_fields: list = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.text_fields = text_fields
        self.output_fields = output_fields
        super().__init__(**kwargs)

    @property
    def name(self):
        return "emotion"
