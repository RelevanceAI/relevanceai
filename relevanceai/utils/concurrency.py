"""Multithreading Module
"""
import json
import math

import threading
import multiprocessing as mp
import uuid

from tqdm.auto import tqdm

from concurrent.futures import (
    as_completed,
    wait,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from relevanceai.constants.constants import HALF_CHUNK_CODES, RETRY_CODES, SUCCESS_CODES
from relevanceai.utils.json_encoder import json_encoder

from relevanceai.utils.progress_bar import NullProgressBar, progress_bar


def chunk(iterables, n=20):
    return [iterables[i : i + n] for i in range(0, int(len(iterables)), int(n))]


def multithread(
    func, iterables, max_workers=2, chunksize=20, show_progress_bar: bool = False
):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        progress_tracker = progress_bar(
            range(math.ceil(len(iterables) / chunksize)),
            show_progress_bar=show_progress_bar,
        )

        futures = [executor.submit(func, it) for it in chunk(iterables, chunksize)]

        if show_progress_bar:
            with progress_tracker as pt:
                for _ in as_completed(futures):
                    if hasattr(pt, "update"):
                        pt.update(1)
        else:
            wait(futures)

        return [future.result() for future in futures]


def multiprocess(
    func,
    iterables,
    max_workers=2,
    chunksize=20,
    post_func_hook: Callable = None,
    show_progress_bar: bool = False,
    process_args: tuple = (),
):
    # with progress_bar(total=int(len(iterables) / chunksize),
    #     show_progress_bar=show_progress_bar) as pbar:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create trackers
        progress_tracker = progress_bar(
            range(math.ceil(len(iterables) / chunksize)),
            show_progress_bar=show_progress_bar,
        )
        progress_iterator = iter(progress_tracker)

        if len(process_args) > 0:
            futures = [
                executor.submit(func, it, process_args)
                for it in chunk(iterables, chunksize)
            ]
        else:
            futures = [executor.submit(func, it) for it in chunk(iterables, chunksize)]
        results = []
        for future in as_completed(futures):
            if post_func_hook:
                results.append(post_func_hook(future.result()))
            else:
                results.append(future.result())
            if progress_tracker is not None:
                next(progress_iterator)
            if show_progress_bar is True:
                progress_tracker.update(1)
        return results


def multiprocess_list(
    func,
    iterables,
    max_workers=2,
    chunksize=1,
    post_func_hook: Callable = None,
    show_progress_bar: bool = False,
    process_args: tuple = (),
):
    # with progress_bar(total=int(len(iterables) / chunksize),
    #     show_progress_bar=show_progress_bar) as pbar:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create trackers
        progress_tracker = progress_bar(
            range(math.ceil(len(iterables) / chunksize)),
            show_progress_bar=show_progress_bar,
        )
        progress_iterator = iter(progress_tracker)

        if len(process_args) > 0:
            futures = [executor.submit(func, it, process_args) for it in iterables]
        else:
            futures = [executor.submit(func, it) for it in iterables]
        results = []
        for future in as_completed(futures):
            if post_func_hook:
                results.append(post_func_hook(future.result()))
            else:
                results.append(future.result())
            if progress_tracker is not None:
                next(progress_iterator)
            if show_progress_bar is True:
                progress_tracker.update(1)
        return results


class Push:
    def __init__(
        self,
        dataset,
        documents: List[Dict[str, Any]],
        batch_size: int,
        max_workers: Optional[int] = None,
        ingest_in_background: bool = True,
        show_progress_bar: bool = True,
        background_execution: bool = False,
        insert_date: bool = True,
        overwrite: bool = True,
        update_schema: bool = True,
        field_transformers: Optional[List] = None,
    ):
        from relevanceai.dataset.dataset import Dataset

        self.dataset: Dataset = dataset
        self.dataset_id: str = dataset.dataset_id

        documents = json_encoder(documents)

        self.frontier = {document["_id"]: 0 for document in documents}
        self.push_queue: mp.Queue = mp.Queue(maxsize=len(documents))
        for document in documents:
            self.push_queue.put(document)

        self.overwrite = overwrite
        self.ingest_in_background = ingest_in_background
        self.batch_size = batch_size
        self.max_workers = 2 if max_workers is None else max_workers
        self.show_progress_bar = show_progress_bar
        self.background_execution = background_execution

        self.lock = threading.Lock()
        self.tqdm_kwargs = dict(leave=True, disable=(not show_progress_bar))
        self.insert_count = 0

        self.insert_date = insert_date
        self.update_schema = update_schema
        self.field_transformers = field_transformers

        self.push_bar = tqdm(
            range(len(documents)),
            desc="push",
            **self.tqdm_kwargs,
        )

    @property
    def failed_ids(self) -> List[str]:
        return [document for document, fails in self.frontier.items() if fails > 0]

    def _get_batch(self):
        batch = []
        while not self.push_queue.empty():
            if len(batch) >= self.batch_size:
                break
            try:
                document = self.push_queue.get()
            except:
                break
            batch.append(document)
        return batch

    def _handle_failed_documents(
        self,
        res: Dict[str, Any],
        batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        failed_documents = res["response_json"]["failed_documents"]
        readded_documents = []

        if failed_documents:
            with self.lock:
                desc = f"push - failed_documents = {self.failed_ids}"
                self.push_bar.set_description(desc)

            # ...find these failed documents within the batch...
            failed_ids = set(map(lambda x: x["_id"], failed_documents))
            failed_documents = [
                document for document in batch if document["_id"] in failed_ids
            ]

            # ...and re add them to the push queue
            for failed_document in failed_documents:
                _id = failed_document["_id"]
                if self.frontier[_id] <= 3:
                    self.frontier[_id] += 1
                    self.push_queue.put(failed_document)
                    readded_documents.append(failed_document)

        return readded_documents

    def _push(self) -> None:

        while True:
            with self.lock:
                batch = self._get_batch()

            if not batch:
                break

            result = self.dataset.datasets.documents.bulk_update(
                dataset_id=self.dataset_id,
                documents=batch,
                insert_date=self.insert_date,
                overwrite=self.overwrite,
                update_schema=self.update_schema,
                field_transformers=self.field_transformers,
                return_documents=True,
                ingest_in_background=self.ingest_in_background,
            )

            failed_documents = self._handle_failed_documents(result, batch)

            inserted = len(batch) - len(failed_documents)

            with self.lock:
                self.insert_count += inserted
                self.push_bar.update(inserted)

    def run(self) -> Tuple[int, List[str]]:
        push_threads = [threading.Thread(target=self._push)]
        for thread in push_threads:
            thread.start()

        if not self.background_execution:
            for thread in push_threads:
                thread.join()

        return self.insert_count, self.failed_ids
