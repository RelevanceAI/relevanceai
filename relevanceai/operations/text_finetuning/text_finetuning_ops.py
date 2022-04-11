"""

.. code-block::

    dataset_id = '???'
    ds = client.Dataset(dataset_id)

    select_fields = [???]
    docs = ds.get_all_documents(select_fields=selected_fields)

    documents = docs[:???] # All of it or the portion that you want to use for fir-tuning

    base_model = 'distilbert-base-uncased' # or other models
    gpl_obj = GPL(base_model = base_model)

    text_field = '???'          # the field you want to use for fine-tuning
    dir_to_save_corpus = "???"  # path to save the corpus that is going to be used by the GPL alg.
    title_field = '???'         # If there is a secondary but related field to be used for fine-tuning, can be none

    gpl_obj.prepare_data_for_finetuning(
        documents= documents,
        text_field=text_field,
        dir_to_save_corpus=dir_to_save_corpus,
        title_field=title_field
    )

    path_to_generated_data = '???'  # some temporary data is saved there
    output_dir = '???'              # where the final model is saved

    gpl_obj.fine_tune(
        path_to_generated_data = path_to_generated_data,
        output_dir=output_dir
    )

    model = gpl_obj.get_finetuned_model()

"""

import json
import logging
from typing import List
try:
    import gpl
except ModuleNotFoundError:
    print("Need to install gpl by running `pip install gpl`.")

class GPL:
    def __init__(
        self,
        base_model: str,
        t5_generator: str = "BeIR/query-gen-msmarco-t5-base-v1",
        retrievers: List[str] = ["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size_gpl: int = 16,
    ):
        self.base_model = base_model
        self.t5_generator = t5_generator
        self.retrievers = retrievers
        self.cross_encoder = cross_encoder
        self.batch_size_gpl = batch_size_gpl
        self.output_path = None

        self.get_gpl()

    def get_gpl(self):
        try:
            import gpl
        except ModuleNotFoundError:
            print("Need to install gpl by running `pip install gpl`.")

    def _get_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError:
            print(
                "Need to install SentenceTransformer by running `pip install sentence_transformers`."
            )

    def prepare_data_for_finetuning(
        self,
        documents: List[dict],
        text_field: str,
        dir_to_save_corpus: str,
        title_field: str = None,
    ):
        # Writes a corpus in the format compatible to the GPL library to later make Pos/Neg pairs
        corpus = []
        i = 0
        with open(f"{dir_to_save_corpus}/corpus.jsonl", "w") as jsonl:
            for doc in documents:
                if text_field not in doc:
                    continue
                line = {
                    "_id": str(i),
                    "title": doc[title_field].replace("\n", " ")
                    if title_field in doc
                    else "",
                    "text": doc[text_field].replace("\n", " "),
                    "metadata": {
                        f: doc[f] for f in doc if f not in [text_field, title_field]
                    },
                }
                # iteratively write lines to the JSON lines corpus.jsonl file
                jsonl.write(json.dumps(line) + "\n")
                i += 1
                corpus.append(line)

        logging.info(
            f"A corpus of {len(corpus)} documents is saved at {dir_to_save_corpus}/corpus.jsonl"
        )

    def fine_tune(
        self,
        path_to_generated_data: str,
        output_dir: str,
        gpl_steps: int = 500,
        do_evaluation: bool = False,
    ):
        # Generates pairs for fine-tuning a model following the GPL algorithm. The final model is saved at  output_dir
        self.path_to_finetuned_model = output_dir
        gpl.train( # type: ignore
            path_to_generated_data=path_to_generated_data,
            base_ckpt=self.base_model,
            batch_size_gpl=self.batch_size_gpl,
            gpl_steps=gpl_steps,
            output_dir=output_dir,
            generator=self.t5_generator,
            retrievers=self.retrievers,
            cross_encoder=self.cross_encoder,
            qgen_prefix="qgen",
            do_evaluation=do_evaluation,
        )

    def get_finetuned_model(self):
        self._get_model()
        if not self.output_path:
            logging.warning("No Fine-Tuned Model Was Found")
        else:
            return SentenceTransformer(self.output_path)

class GPLOps:
    """
    Prepare the models.

    """
    def prepare_data(self):
        raise NotImplementedError
    
    def finetune(self, dataset, field, filters):
        """Finetune on a dataset
        """
        raise NotImplementedError
