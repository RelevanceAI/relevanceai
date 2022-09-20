"""
Base class for base.py to inherit.
All functions related to running operations on datasets.

The Pull Transform Push library is designed to be able to consistently
data from Relevance AI Database, transform and then constantly push data 
to the Relevance AI Database. This ensures that resources are utilised 
to their limits.

"""
import sys
import time
import psutil
import threading
import multiprocessing as mp
import warnings

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Tuple, Type, Union, Optional, Callable
from relevanceai.constants.constants import CONFIG

from relevanceai.dataset import Dataset
from relevanceai.operations_new.transform_base import TransformBase

from tqdm.auto import tqdm

from relevanceai.utils.helpers.helpers import getsizeof

KILL_SIGNAL = "_KILL_QUEUE_SIGNAL_"


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
    _has_kill_signal: bool = True

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        pull_dataset: Optional[Dataset] = None,
        func: Optional[Callable] = None,
        push_dataset: Optional[Dataset] = None,
        func_args: Optional[Tuple[Any, ...]] = None,
        func_kwargs: Optional[Dict[str, Any]] = None,
        multithreaded_update: bool = False,
        pull_chunksize: Optional[int] = None,
        warmup_chunksize: Optional[int] = None,
        transform_chunksize: Optional[int] = 128,
        push_chunksize: Optional[int] = None,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        transform_workers: int = 1,
        push_workers: int = 1,
        buffer_size: int = 0,
        show_progress_bar: bool = True,
        show_pull_progress_bar: bool = True,
        show_transform_progress_bar: bool = True,
        show_push_progress_bar: bool = True,
        ingest_in_background: bool = False,
        run_in_background: bool = False,
        ram_ratio: float = 0.8,
        update_all_at_once: bool = False,
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

        self.update_all_at_once = update_all_at_once
        if update_all_at_once:
            self.transform_chunksize = self.ndocs

        if func is None:
            tqdm.write(f"Transform Chunksize: {self.transform_chunksize:,}")

        self.ingest_in_background = ingest_in_background

        self.filters = [] if filters is None else filters
        self.select_fields = [] if select_fields is None else select_fields

        self.general_lock = threading.Lock()

        self.transform_batch_lock = threading.Lock()
        self.push_batch_lock = threading.Lock()

        self.func_lock: Union[threading.Lock, None]

        if not multithreaded_update:
            self.func_lock = threading.Lock()
            self.transform_workers = 1
        else:
            self.func_lock = None
            self.transform_workers = transform_workers

        self.push_workers = push_workers

        self.func_args = () if func_args is None else func_args
        self.func_kwargs = {} if func_kwargs is None else func_kwargs

        if update_all_at_once:
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
        self.tq: mp.JoinableQueue = mp.JoinableQueue(maxsize=self.single_queue_size)
        self.pq: mp.JoinableQueue = mp.JoinableQueue(maxsize=self.single_queue_size)
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

        # time limit is 1hr 50mins if not set, this is to leave 10mins at the end
        # of sagemaker job to send workflow status email
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
        return sum(document_sizes) / len(sample_documents)

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
        tqdm.write(f"{method.capitalize()} Chunksize: {chunksize}")
        return chunksize

    def _pull(self):
        """
        Iteratively pulls documents from a dataset and places them in the transform queue
        """
        documents: List[Dict[str, Any]] = [{"placeholder": "placeholder"}]
        after_id: Union[List[str], None] = self.after_id

        while not self.timeout_event.is_set():
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
                with self.general_lock:
                    self.ndocs = self.pull_count
                print("Killing transform queue.")
                self.tq.put(KILL_SIGNAL)
                break
            after_id = res["after_id"]

            if self.pull_chunksize is None:
                self.pull_chunksize = self._get_optimal_chunksize(
                    documents[:10], "pull"
                )

            for document in documents:
                self.tq.put(document)

            with self.general_lock:
                self.pull_bar.update(len(documents))
                self.pull_count += len(documents)

    def _get_transform_batch(self) -> List[Dict[str, Any]]:
        """
        Collects a batches of of size `transform_chunksize` from the transform queue
        """
        batch: List[Dict[str, Any]] = []

        queue = self.tq
        timeout = 5 if not self.update_all_at_once else None

        if self.transform_count == 0 and self.warmup_chunksize is not None:
            chunksize = self.warmup_chunksize
            tqdm.write("Processing Warmup Batch")
        else:
            chunksize = self.transform_chunksize

        while len(batch) < chunksize:
            try:
                document = queue.get(timeout=timeout)
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
        timeout = 5

        # Calculate optimal batch size
        if self.push_chunksize is None:
            sample_documents = []
            for _ in range(10):
                try:
                    sample_document = queue.get(timeout=timeout)
                except:
                    break
                sample_documents.append(sample_document)

            self.push_chunksize = self._get_optimal_chunksize(sample_documents, "push")
            batch = sample_documents

        chunksize = self.push_chunksize
        while len(batch) < chunksize:
            try:
                document = queue.get(timeout=timeout)
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
        # Check for early termination (such as no documents)
        HAS_KILL_SIGNAL: bool = False
        print("Begin transform.")
        while self.transform_count <= self.ndocs and not self.timeout_event.is_set():
            print("Inside transform loop.")
            with self.transform_batch_lock:
                batch = self._get_transform_batch()
                if len(batch) > 0:
                    if batch[-1] == KILL_SIGNAL:
                        HAS_KILL_SIGNAL = True
                        print("Killing transform queue.")
                        self.tq.task_done()
                        batch = batch[:-1]

            if self.func is not None:
                old_batch = deepcopy(batch)

                if self.func_lock is not None:
                    with self.func_lock:
                        try:
                            new_batch = self.func(
                                batch, *self.func_args, **self.func_kwargs
                            )
                        except Exception as e:
                            print(e)
                            new_batch = batch
                else:
                    try:
                        new_batch = self.func(batch, **self.func_kwargs)
                    except Exception as e:
                        print(e)
                        new_batch = batch

                batch = PullTransformPush._postprocess(new_batch, old_batch)

            for document in batch:
                self.pq.put(document)

            # Send kill signal to push queue
            if HAS_KILL_SIGNAL:
                print("Killing Push queue")
                self.pq.put(KILL_SIGNAL)

            with self.general_lock:
                self.transform_bar.update(len(batch))
                self.transform_count += len(batch)
            if HAS_KILL_SIGNAL:
                sys.exit()

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
            with self.general_lock:
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
        # Ensure kill signal is only sent after the transformation
        # is done and the push is added
        HAS_KILL_SIGNAl = False
        while self.push_count < self.ndocs and not self.timeout_event.is_set():
            with self.push_batch_lock:
                batch = self._get_push_batch()
                if len(batch) > 0:
                    if batch[-1] == KILL_SIGNAL:
                        HAS_KILL_SIGNAl = True
                        batch = batch[:-1]
                        self.pq.task_done()

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

            failed_documents = self._handle_failed_documents(res, batch)

            with self.general_lock:
                self.push_bar.update(len(batch) - len(failed_documents))
                self.push_count += len(batch) - len(failed_documents)

            # tell push queue to also finish
            if HAS_KILL_SIGNAl:
                self.pq.task_done()
                # Close threads
                sys.exit()

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
        self.pull_thread = threading.Thread(target=self._pull, daemon=daemon)
        self.transform_threads = [
            threading.Thread(target=self._transform, daemon=daemon)
            for _ in range(self.transform_workers)
        ]
        self.push_threads = [
            threading.Thread(target=self._push, daemon=daemon)
            for _ in range(self.push_workers)
        ]

    def _start_worker_threads(self):
        """
        Start the worker threads
        """

        # Start fetching data from the server
        self.pull_thread.start()
        # Once there is data in the queue, then we start the
        # transform threads
        while self.tq.empty():
            time.sleep(1)
        for thread in self.transform_threads:
            thread.start()
        while self.pq.empty():
            time.sleep(1)
        for thread in self.push_threads:
            thread.start()

    def _join_worker_threads(self, timeout: Optional[int] = None):
        """
        ...and then join them in reversed order.
        """

        # Try to join if not running in background
        if not self.run_in_background:
            self.pull_thread.join(timeout=timeout)
            for thread in self.transform_threads:
                thread.join(timeout=timeout)
            for thread in self.push_threads:
                thread.join(timeout=timeout)

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
                tqdm.write("Time Limit Exceeded")
                with self.general_lock:
                    self.timeout_event.set()
                tqdm.write("Exiting Operation...")
                break

            # or if all the threads have finished
            if not self._threads_are_alive():
                break

            # poll these checks every 1 sec
            time.sleep(1.0)

    def _flush_queues(self, timeout: float = 1e-2):
        """
        Gets all items in both queues to avoid
        BrokenPipeError when calling queue.close()
        """
        while True:
            try:
                self.tq.get(timeout=timeout)
            except:
                break

        while True:
            try:
                self.pq.get(timeout=timeout)
            except:
                break

        self.tq.close()
        self.pq.close()

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

            if self.timeout is None:
                self._join_worker_threads()

            else:
                self._run_timer()  # Starts the timer
                self._flush_queues()
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
        batched: Optional[bool] = False,
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
                update_all_at_once=(not batched),
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
        transform_workers: int = 2,
        push_workers: int = 2,
        timeout: int = None,
        buffer_size: int = 0,
        show_progress_bar: bool = True,
        warmup_chunksize: int = None,
        transform_chunksize: int = 32,
        multithreaded_update: bool = False,
        update_all_at_once: bool = False,
        ingest_in_background: bool = True,
        **kwargs,
    ):
        if multithreaded_update:
            warnings.warn(
                "Multithreaded-update should be False for vectorizing with 1 GPU only. Could hang if True. Works fine on CPU."
            )
        ptp = PullTransformPush(
            dataset=dataset,
            func=self.transform,
            func_args=func_args,
            func_kwargs=func_kwargs,
            multithreaded_update=multithreaded_update,
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
            timeout=timeout,
            update_all_at_once=update_all_at_once,
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
