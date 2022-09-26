"""
Base class for base.py to inherit.
All functions related to running operations on datasets.

The Pull Transform Push library is designed to be able to consistently
data from Relevance AI Database, transform and then constantly push data
to the Relevance AI Database. This ensures that resources are utilised
to their limits.

"""
import math
import os
import sys
import time
import psutil
import warnings
import threading
import traceback
import logging

from queue import Queue
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Tuple, Type, Union, Optional, Callable

from relevanceai.constants.constants import CONFIG
from relevanceai.dataset import Dataset
from relevanceai.operations_new.transform_base import TransformBase
from relevanceai.utils.helpers.helpers import getsizeof

from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


class PullTransformPush:

    pull_count: int
    transform_count: int
    push_count: int

    pull_bar: tqdm
    transform_bar: tqdm
    push_bar: tqdm

    pull_thread: threading.Thread
    transform_threads: List[threading.Thread]
    push_threads: List[threading.Thread]

    pull_dataset: Dataset
    push_dataset: Dataset

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        pull_dataset: Optional[Dataset] = None,
        push_dataset: Optional[Dataset] = None,
        func: Optional[Callable] = None,
        func_args: Optional[Tuple[Any, ...]] = None,
        func_kwargs: Optional[Dict[str, Any]] = None,
        pull_chunksize: Optional[int] = None,
        push_chunksize: Optional[int] = None,
        transform_chunksize: Optional[int] = 128,
        warmup_chunksize: Optional[int] = None,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        transform_workers: Optional[int] = None,
        push_workers: Optional[int] = None,
        buffer_size: int = 0,
        show_progress_bar: bool = True,
        show_pull_progress_bar: bool = True,
        show_transform_progress_bar: bool = True,
        show_push_progress_bar: bool = True,
        ingest_in_background: bool = True,
        run_in_background: bool = False,
        ram_ratio: float = 0.8,
        batched: bool = False,
        retry_count: int = 3,
        after_id: Optional[List[str]] = None,
        pull_limit: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        """
        Buffer size:
            number of documents to be held in limbo by both queues at any one time

        """
        super().__init__()

        if dataset is None:
            self.push_dataset = push_dataset  # type: ignore
            self.pull_dataset = pull_dataset  # type: ignore

        else:
            self.push_dataset = dataset
            self.pull_dataset = dataset

        if dataset is None and pull_dataset is None and push_dataset is None:
            raise ValueError(
                "Please set `dataset=` or `push_dataset=` and `pull_dataset=`"
            )

        self.pull_dataset_id = self.pull_dataset.dataset_id
        self.push_dataset_id = self.push_dataset.dataset_id

        self.config = CONFIG

        self.pull_limit = pull_limit

        self.failed_documents_count = 0

        if pull_limit is None:
            ndocs = self.pull_dataset.get_number_of_documents(
                dataset_id=self.pull_dataset_id,
                filters=filters,
            )
            self.ndocs = ndocs
        else:
            self.ndocs = pull_limit

        self.pull_chunksize = pull_chunksize
        self.transform_chunksize = min(transform_chunksize, self.ndocs)
        self.warmup_chunksize = warmup_chunksize
        self.push_chunksize = push_chunksize

        self.batched = batched
        if not batched:
            self.transform_chunksize = self.ndocs

        if func is None:
            tqdm.write(f"Transform Chunksize: {self.transform_chunksize:,}")

        self.ingest_in_background = ingest_in_background

        self.filters = [] if filters is None else filters
        self.select_fields = [] if select_fields is None else select_fields

        self.func_lock: Union[threading.Lock, None]

        cpu_count = os.cpu_count() or 1

        if batched:
            self.transform_workers = (
                math.ceil(cpu_count / 4)
                if transform_workers is None
                else transform_workers
            )
        else:
            self.transform_workers = 1

        msg = f"Using {self.transform_workers} transform worker(s)"
        tqdm.write(f"Using {self.transform_workers} transform worker(s)")
        logger.debug(msg)

        self.push_workers = (
            math.ceil(cpu_count / 4) if push_workers is None else push_workers
        )
        msg = f"Using {self.push_workers} push worker(s)"
        tqdm.write(f"Using {self.push_workers} push worker(s)")
        logger.debug(msg)

        self.func_args = () if func_args is None else func_args
        self.func_kwargs = {} if func_kwargs is None else func_kwargs

        if not batched:
            self.single_queue_size = self.ndocs

        else:
            if buffer_size == 0:
                ram_size = psutil.virtual_memory().total  # in bytes

                # assuming documents are 1MB, this is an upper limit and accounts for alot
                average_size = 2**20
                max_document_size = min(average_size, 2**20)

                total_queue_size = int(ram_size * ram_ratio / max_document_size)
            else:
                total_queue_size = buffer_size

            self.single_queue_size = int(total_queue_size / 2)

        tqdm.write(f"Max number of documents in queue: {self.single_queue_size:,}")

        # mp queues are thread-safe while being able to be used across processes
        self.tq: Queue = Queue(maxsize=self.single_queue_size)
        self.pq: Queue = Queue(maxsize=self.single_queue_size)
        self.func = func

        self.pull_tqdm_kwargs = dict(
            leave=True,
            disable=(not (show_pull_progress_bar and show_progress_bar)),
        )
        if not self.pull_tqdm_kwargs["disable"]:
            self.pull_tqdm_kwargs["position"] = 0  # type: ignore

        self.transform_tqdm_kwargs = dict(
            leave=True,
            disable=(not (show_transform_progress_bar and show_progress_bar)),
        )
        if not self.transform_tqdm_kwargs["disable"]:
            self.transform_tqdm_kwargs["position"] = (
                self.pull_tqdm_kwargs.get("position", 0) + 1  # type: ignore
            )

        self.push_tqdm_kwargs = dict(
            leave=True,
            disable=(not (show_push_progress_bar and show_progress_bar)),
        )
        if not self.push_tqdm_kwargs["disable"]:
            self.push_tqdm_kwargs["position"] = (
                self.pull_tqdm_kwargs.get("position", 0)  # type: ignore
                + self.transform_tqdm_kwargs.get("position", 0)
                + 1
            )

        self.run_in_background = run_in_background

        self.failed_frontier: Dict[str, int] = {}
        self.retry_count = retry_count
        self.after_id = after_id

        # time limit is infinite if not set. For workflows, this should be ~1hr 50mins = 110mins
        # this is to leave 10mins at the end of sagemaker job to send workflow status email
        self.timeout = timeout
        self.timeout_event = threading.Event()

    def _get_average_document_size(self, sample_documents: List[Dict[str, Any]]):
        """
        Get average size of a document in memory.
        Returns size in bytes.
        """
        document_sizes = [
            getsizeof(sample_document) for sample_document in sample_documents
        ]
        return max(1, sum(document_sizes) / len(sample_documents))

    def _get_optimal_chunksize(
        self, sample_documents: List[Dict[str, Any]], method: str
    ) -> int:
        """
        Calculates the optimal batch size given a list of sampled documents and constraints in config
        """
        document_size = self._get_average_document_size(sample_documents)
        document_size = document_size / 2**20
        target_chunk_mb = int(self.config.get_option("upload.target_chunk_mb"))
        max_chunk_size = int(self.config.get_option("upload.max_chunk_size"))
        chunksize = int(target_chunk_mb / document_size)
        chunksize = min(chunksize, max_chunk_size)

        thread_name = threading.current_thread().name
        msg = f"{thread_name}\tchunksize: {chunksize}".expandtabs(5)
        logger.debug(msg)
        tqdm.write(msg)

        return chunksize

    def _pull(self):
        """
        Iteratively pulls documents from a dataset and places them in the transform queue
        """
        documents: List[Dict[str, Any]] = [{"placeholder": "placeholder"}]
        after_id: Union[List[str], None] = self.after_id

        while not self.timeout_event.is_set():
            logger.debug("pull")

            current_count = self.pull_bar.n

            if self.pull_chunksize is None:
                # this is the first batch
                pull_chunksize = 20
            else:
                # optimized pull batch size (every other pull)
                pull_chunksize = self.pull_chunksize

            if self.pull_limit is None:
                # Very large number if no limits
                pull_limit = sys.maxsize
            else:
                # if there is a limit, get the ndocs left
                pull_limit = self.pull_limit - current_count

            # Consider all scenarios
            # let 3333 = pull_limit, current_count = 3000
            #         = min(20,            sys.maxsize)             = 20
            #         = min(20,            (3333 - 3000 = 333))     = 20
            #         = min(~512,          sys.maxsize)             = 512
            #         = min(~512,          (3333 - 3000 = 333))     = 333
            page_size = min(pull_chunksize, pull_limit)

            res = self.pull_dataset.datasets.documents.get_where(
                dataset_id=self.pull_dataset_id,
                page_size=page_size,
                filters=self.filters,
                select_fields=self.select_fields,
                sort=[],
                after_id=after_id,
            )

            documents = res["documents"]
            if not documents:
                self.ndocs = self.pull_count
                break
            after_id = res["after_id"]

            if self.pull_chunksize is None:
                self.pull_chunksize = self._get_optimal_chunksize(
                    documents[:10], "pull"
                )

            for document in documents:
                self.tq.put(document)

            self.pull_bar.update(len(documents))
            self.pull_count += len(documents)

    def _get_transform_batch(self) -> List[Dict[str, Any]]:
        """
        Collects a batches of of size `transform_chunksize` from the transform queue
        """
        batch: List[Dict[str, Any]] = []

        queue = self.tq

        if self.transform_count == 0 and self.warmup_chunksize is not None:
            chunksize = self.warmup_chunksize
            tqdm.write("Processing Warmup Batch")
        else:
            chunksize = self.transform_chunksize

        while len(batch) < chunksize:
            try:
                if self.batched:
                    document = queue.get_nowait()
                else:
                    document = queue.get()
                batch.append(document)
            except:
                break

        return batch

    def _get_push_batch(self) -> List[Dict[str, Any]]:
        """
        Collects a batches of of size `push_chunksize` from the transform queue
        """
        batch: List[Dict[str, Any]] = []

        queue = self.pq

        # Calculate optimal batch size
        if self.push_chunksize is None:
            sample_documents = []
            for _ in range(10):
                try:
                    sample_document = queue.get(timeout=1)
                except:
                    break
                sample_documents.append(sample_document)

            self.push_chunksize = self._get_optimal_chunksize(sample_documents, "push")
            batch = sample_documents

        chunksize = self.push_chunksize
        while len(batch) < chunksize:
            try:
                document = queue.get_nowait()
                batch.append(document)
            except:
                break

        return batch

    def _transform(self):
        """
        Updates a batch of documents given an update function.
        After updating, remove all fields that are present in both old and new batches.
        ^ necessary to avoid reinserting stuff that is already in the cloud.
        Then, repeatedly put each document from the processed batch in the push queue
        """

        while self.transform_count < self.ndocs and not self.timeout_event.is_set():
            batch = self._get_transform_batch()
            if not batch:
                continue

            try:
                if self.func is not None:
                    old_batch = deepcopy(batch)

                    new_batch = self.func(
                        batch,
                        *self.func_args,
                        **self.func_kwargs,
                    )
                    logger.debug("transformed batch")

                    batch = PullTransformPush._postprocess(new_batch, old_batch)
                    logger.debug("postprocessed batch")

                for document in batch:
                    self.pq.put(document)

            except Exception as e:
                traceback.print_exc()
                print(e)

            self.transform_bar.update(len(batch))
            self.transform_count += len(batch)

    def _handle_failed_documents(
        self,
        res: Dict[str, Any],
        batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Handles any documents that failed to upload.
        Does so by collecting by `_id` from the batch, and reinserting them in the push queue.
        """
        # check if there is any failed documents...
        failed_documents = res["response_json"].get("failed_documents", [])

        if failed_documents:
            self.failed_documents_count += len(failed_documents)
            desc = f"push - failed_documents = {self.failed_documents_count}"
            self.push_bar.set_description(desc)

            # ...find these failed documents within the batch...
            failed_ids = set(map(lambda x: x["_id"], failed_documents))
            failed_documents = [
                document for document in batch if document["_id"] in failed_ids
            ]

            # ...and re add them to the push queue...
            for failed_document in failed_documents:
                _id = failed_document["_id"]
                if _id not in self.failed_frontier:
                    self.failed_frontier[_id] = 0

                # ...only if they have failed less than the retry count
                if self.failed_frontier[_id] < self.retry_count:
                    self.failed_frontier[_id] += 1
                    self.pq.put(failed_document)

        return failed_documents

    @staticmethod
    def _postprocess(
        new_batch: List[Dict[str, Any]],
        old_batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Removes fields from `new_batch` that are present in the `old_keys` list.
        Necessary to avoid bloating the upload payload with unnecesary information.
        """
        batch = []
        for old_document, new_document in zip(old_batch, new_batch):
            document: Dict[str, Any] = {}
            new_fields = Dataset.list_doc_fields(doc=new_document)
            old_fields = Dataset.list_doc_fields(doc=old_document)
            for field in new_fields:
                old_value = Dataset.get_field(
                    field=field,
                    doc=old_document,
                    missing_treatment="return_none",
                )
                new_value = Dataset.get_field(
                    field=field,
                    doc=new_document,
                    missing_treatment="return_none",
                )
                value_diff = old_value != new_value
                if field not in old_fields or value_diff or field == "_id":
                    Dataset.set_field(field=field, doc=document, value=new_value)
            batch.append(document)

        return batch

    @staticmethod
    def _get_updates(batch) -> bool:
        updates = sum(
            [
                len([key for key in document.keys() if key != "_id"])
                for document in batch
            ]
        )
        return True if updates > 0 else False

    def _push(self) -> None:
        """
        Iteratively gather a batch of processed documents and push these to cloud
        """
        while self.push_count < self.ndocs and not self.timeout_event.is_set():
            batch = self._get_push_batch()
            if not batch:
                continue

            try:
                batch = self.pull_dataset.json_encoder(batch)
                update = PullTransformPush._get_updates(batch)

                if update:
                    res = self.push_dataset.datasets.documents.bulk_update(
                        self.push_dataset_id,
                        batch,
                        return_documents=True,
                        ingest_in_background=self.ingest_in_background,
                    )
                else:
                    res = {
                        "response_json": {},
                        "documents": batch,
                        "status_code": 200,
                    }
                    logger.debug("pushed batch")

            except Exception as e:
                traceback.print_exc()
                print(e)

            failed_documents = self._handle_failed_documents(res, batch)

            self.push_bar.update(len(batch) - len(failed_documents))
            self.push_count += len(batch) - len(failed_documents)

    def _init_progress_bars(self) -> None:
        """
        Initialise the progress bars for dispay progress on pulling updating and pushing.
        """
        self.pull_count = 0
        self.pull_bar = tqdm(
            desc="pull",
            total=self.ndocs,
            **self.pull_tqdm_kwargs,
        )
        self.transform_count = 0
        self.transform_bar = tqdm(
            range(self.ndocs),
            desc="transform",
            **self.transform_tqdm_kwargs,
        )
        self.push_count = 0
        self.push_bar = tqdm(
            range(self.ndocs),
            desc="push",
            **self.push_tqdm_kwargs,
        )

    def _init_worker_threads(self) -> None:
        """
        Initialise the worker threads for each process
        """
        daemon = True if self.timeout is not None else False
        self.pull_thread = threading.Thread(
            target=self._pull,
            name="Pull_Worker",
            daemon=daemon,
        )
        self.transform_threads = [
            threading.Thread(
                target=self._transform,
                name=f"Transform_Worker_{index}",
                daemon=daemon,
            )
            for index in range(self.transform_workers)
        ]
        self.push_threads = [
            threading.Thread(
                target=self._push,
                name=f"Push_Worker_{index}",
                daemon=daemon,
            )
            for index in range(self.push_workers)
        ]

    def _start_worker_threads(self):
        """
        Start the worker threads
        """

        # Start fetching data from the server
        self.pull_thread.start()
        logger.debug("started pull thread")

        # Once there is data in the queue, then we start the
        # transform threads
        while True:
            if not self.tq.empty():
                for thread in self.transform_threads:
                    thread.start()
                    logger.debug("started transform thread")

                break
            time.sleep(1)

        while True:
            if not self.pq.empty():
                for thread in self.push_threads:
                    thread.start()
                    logger.debug("started push thread")

                break
            time.sleep(1)

    def _join_worker_threads(self, timeout: Optional[int] = None):
        """
        ...and then join them in order.
        """

        # Try to join if not running in background
        self.pull_thread.join(timeout=timeout)
        logger.debug("joined pull thread")

        for thread in self.transform_threads:
            thread.join(timeout=timeout)
            logger.debug("joined transform thread")

        for thread in self.push_threads:
            thread.join(timeout=timeout)
            logger.debug("joined push thread")

    def _threads_are_alive(self) -> True:
        """
        Poll all active trheads to check if they are alive
        """
        if self.pull_thread.is_alive():
            return True
        for thread in self.transform_threads:
            if thread.is_alive():
                return True
        for thread in self.push_threads:
            if thread.is_alive():
                return True
        return False

    def _run_timer(self):
        start_time = time.time()
        while True:
            current_time = time.time()

            # check if time limit was exceeded
            if (current_time - start_time) >= self.timeout:
                self.timeout_event.set()

                msg = "Time Limit Exceeded\nExiting Operation..."
                tqdm.write(msg)
                logger.debug(msg)
                break

            # or if all the threads have finished
            if not self._threads_are_alive():
                break

            # poll these checks every 1 sec
            time.sleep(1.0)

    def run(self) -> Dict[str, Any]:
        """
        (Main Method)
        Do the pulling, the updating, and of course, the pushing.

        return the _ids of any failed documents
        """

        if self.ndocs > 0:
            self._init_progress_bars()
            self._init_worker_threads()
            self._start_worker_threads()

            if self.timeout is None and not self.run_in_background:
                self._join_worker_threads()

            else:
                self._run_timer()  # Starts the timer
                # no need to join threads as they are daemons
                # if there is a timeout

        return {
            "timed_out": self.timeout_event.is_set(),
            "failed_documents": list(self.failed_frontier.keys()),
        }


def arguments(cls: Type[PullTransformPush]):
    import inspect

    sig = inspect.signature(cls.__init__)
    return list(sig.parameters)


class OperationRun(TransformBase):
    """
    All functions related to running transforms as an operation on datasets
    """

    def is_chunk_valid(self, chunk):
        return chunk is not None and len(chunk) > 0

    # @abstractmethod
    def post_run(self, dataset, documents, updated_documents):
        pass

    def run(
        self,
        dataset: Dataset,
        batched: bool = False,
        chunksize: Optional[int] = None,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        output_fields: Optional[list] = None,
        refresh: bool = False,
        **kwargs,
    ):
        """It takes a dataset, and then it gets all the documents from that dataset. Then it transforms the
        documents and then it upserts the documents.

        Parameters
        ----------
        dataset : Dataset
            Dataset,
        select_fields : list
            Used to determine which fields to retrieve for filters
        output_fields: list
            Used to determine which output fields are missing to continue running operation

        filters : list
            list = None,

        """

        if filters is None:
            filters = []
        if select_fields is None:
            select_fields = []

        # store this
        if hasattr(dataset, "dataset_id"):
            self.dataset_id = dataset.dataset_id

        schema = dataset.schema
        self._check_fields_in_schema(schema=schema, fields=select_fields)

        filters += [
            {
                "filter_type": "or",
                "condition_value": [
                    {
                        "field": field,
                        "filter_type": "exists",
                        "condition": "==",
                        "condition_value": " ",
                    }
                    for field in select_fields
                ],
            }
        ]

        # add a checkmark for output fields
        if not refresh and output_fields is not None and len(output_fields) > 0:
            filters += [
                {
                    "field": output_fields[0],
                    "filter_type": "exists",
                    "condition": "!=",
                    "condition_value": " ",
                }
            ]

        # needs to be here due to circular imports
        from relevanceai.operations_new.ops_manager import OperationManager

        with OperationManager(
            dataset=dataset,
            operation=self,
        ) as dataset:
            res = self.batch_transform_upsert(
                dataset=dataset,
                select_fields=select_fields,
                filters=filters,
                chunksize=chunksize,
                batched=batched,
                **kwargs,
            )

        return res

    def batch_transform_upsert(
        self,
        dataset: Dataset,
        func_args: Optional[Tuple[Any]] = None,
        func_kwargs: Optional[Dict[str, Any]] = None,
        select_fields: list = None,
        filters: list = None,
        chunksize: int = None,
        transform_workers: Optional[int] = None,
        push_workers: Optional[int] = None,
        buffer_size: int = 0,
        show_progress_bar: bool = True,
        warmup_chunksize: int = None,
        transform_chunksize: int = 128,
        batched: bool = False,
        ingest_in_background: bool = True,
        **kwargs,
    ):
        ptp = PullTransformPush(
            dataset=dataset,
            func=self.transform,
            func_args=func_args,
            func_kwargs=func_kwargs,
            pull_chunksize=chunksize,
            warmup_chunksize=warmup_chunksize,
            transform_chunksize=transform_chunksize,
            push_chunksize=chunksize,
            filters=filters,
            select_fields=select_fields,
            transform_workers=transform_workers,
            push_workers=push_workers,
            buffer_size=buffer_size,
            show_progress_bar=show_progress_bar,
            batched=batched,
            ingest_in_background=ingest_in_background,
            **kwargs,
        )
        ptp.run()

    def store_operation_metadata(
        self,
        dataset: Dataset,
        values: Optional[Dict[str, Any]] = None,
    ):
        """This function stores metadata about operators

        Parameters
        ----------
        dataset : Dataset
            Dataset,
        values : Optional[Dict[str, Any]]
            Optional[Dict[str, Any]] = None,

        Returns
        -------
            The dataset object with the metadata appended to it.

        .. code-block::

            {
                "_operationhistory_": {
                    "1-1-1-17-2-3": {
                        "operation": "vector",
                        "model_name": "miniLm"
                    },
                }
            }

        """
        if values is None:
            values = self.get_operation_metadata()

        tqdm.write("Storing operation metadata...")
        timestamp = str(datetime.now().timestamp()).replace(".", "-")
        metadata = dataset.metadata.to_dict()
        if "_operationhistory_" not in metadata:
            metadata["_operationhistory_"] = {}
        metadata["_operationhistory_"].update(
            {
                timestamp: {
                    "operation": self.name,
                    "parameters": str(values),
                }
            }
        )
        # Gets metadata and appends to the operation history
        return dataset.upsert_metadata(metadata)
