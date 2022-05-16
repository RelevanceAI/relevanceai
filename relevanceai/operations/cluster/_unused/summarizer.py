from typing import Dict, List

from relevanceai.utils import DocUtils
from relevanceai.utils.logger import LoguruLogger

from relevanceai.constants.errors import MissingPackageError


class TransformersLMSummarizer(LoguruLogger, DocUtils):
    """Seq2seq models using Summarizer class"""

    def __init__(self, model: str, tokenizer: str, *args, **kwargs):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ModuleNotFoundError as e:
            raise MissingPackageError("transformers")
        try:
            import torch

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except ModuleNotFoundError as e:
            raise MissingPackageError("torch")

        if not any([f in model for f in ["t5", "bart"]]):
            raise ValueError(
                "Model must be of t5 or  bart base.\n \
                The models that this pipeline can use are models that have been fine-tuned on a summarization task, \
                which is currently, '`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'. \
                See the up-to-date list of available models on \
                    `huggingface.co/models <https://huggingface.co/models?filter=summarization>`__."
            )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __call__(self, text: str, max_length: int = 100, num_beams: int = 4):
        self.model = self.model.to(self.device)

        # tokenize without truncation
        inputs_no_trunc = self.tokenizer(
            text, max_length=None, return_tensors="pt", truncation=False
        ).to(self.device)

        # get batches of tokens corresponding to the exact model_max_length
        chunk_start = 0
        chunk_end = self.tokenizer.model_max_length
        inputs_batch_l = []
        while chunk_start <= len(inputs_no_trunc["input_ids"][0]):
            inputs_batch = inputs_no_trunc["input_ids"][0][
                chunk_start:chunk_end
            ]  # get batch of n tokens
            inputs_batch = inputs_batch.reshape(
                (1,) + inputs_batch.shape
            )  ## add batch dimension
            inputs_batch_l.append(inputs_batch)
            chunk_start += self.tokenizer.model_max_length
            chunk_end += self.tokenizer.model_max_length

        # generate a summary on each batch
        summary_ids_l = [
            self.model.generate(
                inputs.to(self.device),
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True,
            ).to(self.device)
            for inputs in inputs_batch_l
        ]

        # decode the output and join into one string with one paragraph per summary batch
        summary_batch_l = []
        for summary_id in summary_ids_l:
            summary_batch = [
                self.tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for g in summary_id
            ]
            summary_batch_l.append(summary_batch[0])

        return [{"summary_text": " ".join(summary_batch_l).replace(" .", ".").strip()}]
