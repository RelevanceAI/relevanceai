from relevanceai.package_utils.logger import LoguruLogger
from doc_utils import DocUtils


try:
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{e}\nInstall torch\n \
        pip install -U torch"
    )


class TransformersLMSummarizer(LoguruLogger, DocUtils):
    def __init__(self, model, tokenizer, *args, **kwargs):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\nInstall transformers\n \
                pip install -U transformers"
            )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __call__(self, text):
        # tokenize without truncation
        inputs_no_trunc = self.tokenizer(
            text, max_length=None, return_tensors="pt", truncation=False
        )

        # get batches of tokens corresponding to the exact model_max_length
        chunk_start = 0
        chunk_end = self.tokenizer.model_max_length
        inputs_batch_l = []
        while chunk_start <= len(inputs_no_trunc["input_ids"][0]):
            inputs_batch = inputs_no_trunc["input_ids"][0][
                chunk_start:chunk_end
            ]  # get batch of n tokens
            inputs_batch = torch.unsqueeze(inputs_batch, 0)
            inputs_batch_l.append(inputs_batch)
            chunk_start += self.tokenizer.model_max_length
            chunk_end += self.tokenizer.model_max_length

        # generate a summary on each batch
        summary_ids_l = [
            self.model.generate(
                inputs, num_beams=4, max_length=100, early_stopping=True
            ).to(device)
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
