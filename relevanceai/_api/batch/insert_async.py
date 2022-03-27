import asyncio
import time
import traceback
import warnings

import orjson

from datetime import datetime
from concurrent.futures import as_completed, wait
from pathlib import Path
from threading import Thread
from typing import Callable, Optional, Tuple

from appdirs import user_cache_dir
from relevanceai._api.batch.retrieve import BatchRetrieveClient
from relevanceai._api.endpoints.api_client import APIEndpointsClient
from relevanceai.utils.logger import FileLogger
from relevanceai.utils.progress_bar import progress_bar
from relevanceai.constants.warning import Warning

MB_TO_BYTE = 1024 * 1024


class EventLoop(Thread):
    """
    The event loop is required to call asynchronous functions from
    a regular (synchronous) function. This class spins off a
    background thread that runs an event loop. All asynchronous
    tasks are run there.
    """

    def __init__(self, tasks, show_progress_bar):
        super().__init__()
        self._loop = asyncio.new_event_loop()
        self.daemon = True

        self.futures = []
        self.tasks = tasks

        self.show_progress_bar = show_progress_bar
        self.progress_bar = progress_bar(
            range(len(tasks)), show_progress_bar=show_progress_bar
        )

    def run(self):
        self._loop.run_forever()

    async def _create_future(self, task):
        return await asyncio.create_task(task)

    def execute_tasks(self):
        self.futures.extend(
            [
                asyncio.run_coroutine_threadsafe(self._create_future(task), self._loop)
                for task in self.tasks
            ]
        )

        if self.show_progress_bar:
            with self.progress_bar as progress_bar:
                for _ in as_completed(self.futures):
                    progress_bar.update(1)
        else:
            wait(self.futures)

    def terminate(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self.join()


class BatchInsertAsyncHelpers(BatchRetrieveClient, APIEndpointsClient):
    async def _apply_bulk_fn(self, bulk_fn, documents: list):
        """
        Called from _process_documents. Calls bulk_fn on documents.

        Parameters
        ----------
        bulk_fn
            An asynchronous function that takes in the documents

        documents: list
            A list of documents
        """
        if len(documents) == 0:
            warnings.warn(Warning.NO_DOCUMENT_DETECTED)
            return {
                "inserted": 0,
                "failed_documents": [],
                "failed_documents_detailed": [],
            }

        num_documents_inserted: int = 0
        # Maintain a reference to documents to keep track during looping
        documents_remaining = documents.copy()
        for _ in range(int(self.config.get_option("retries.number_of_retries"))):
            if len(documents_remaining) > 0:
                # bulk_update_async
                response = await bulk_fn(documents=documents)
                num_documents_inserted += response["inserted"]
                documents_remaining = [
                    document
                    for document in documents_remaining
                    if document["_id"] in response["failed_documents"]
                ]
            else:
                # Once documents_remaining is empty, leave the for-loop...
                break
            # ...else, wait some amount of time before retrying
            time.sleep(int(self.config["retries.seconds_between_retries"]))

        return {
            "inserted": num_documents_inserted,
            "failed_documents": documents_remaining,
        }

    async def _process_documents(
        self,
        dataset_id: str,
        documents: list,
        insert: bool,
        use_json_encoder: bool = True,
        create_id: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Called from pull_update_push_async. This method determines via user
        input whether to insert or to update documents in a Dataset. (A
        Dataset that does not exist is automatically created.) Then, the
        operation (either bulk_insert_async or bulk_update_async) is wrapped
        by _apply_bulk_fn and then applied to the given documents.

        Parameters
        ----------
        dataset_id: str
            The dataset_id of the collection to change

        documents: list
            A list of documents

        insert: bool
            If True, inserts rather than updates an already-existing dataset

        use_json_encoder: bool
            Whether to automatically convert documents to json encodable format

        create_id: bool
            If True, creates a indices for the documents

        verbose: bool
            If True, print user-informing statements.

        **kwargs
            Additional arguments for bulk_insert_async or bulk_update_async
        """
        if use_json_encoder:
            documents = self.json_encoder(documents)

        in_dataset = dataset_id in self.datasets.list()["datasets"]
        if not in_dataset or insert:
            operation = f"inserting into {dataset_id}"
            if not in_dataset:
                self.datasets.create(dataset_id)
            if insert:
                create_id = True
            self._convert_id_to_string(documents, create_id=create_id)

            async def bulk_fn(documents):
                return await self.datasets.bulk_insert_async(
                    dataset_id=dataset_id,
                    documents=documents,
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if key not in {"dataset_id", "updates"}
                    },
                )

        else:
            operation = f"updating {dataset_id}"
            self._convert_id_to_string(documents, create_id=create_id)

            async def bulk_fn(documents):
                return await self.datasets.documents.bulk_update_async(
                    dataset_id=dataset_id,
                    updates=documents,
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if key not in {"dataset_id", "documents"}
                    },
                )

        self.logger.info(f"You are currently {operation}")

        self.logger.info(
            "You can track your stats and progress via our dashboard at "
            + f"https://cloud.relevance.ai/collections/dashboard/stats/?collection={dataset_id}"
        )

        if verbose:
            print(
                f"While {operation}, you can visit your dashboard at "
                + f"https://cloud.relevance.ai/dataset/{dataset_id}/dashboard/monitor/"
            )

        return await self._apply_bulk_fn(bulk_fn=bulk_fn, documents=documents)

    def _get_avg_document_size(
        self,
        dataset_id: str,
        filters: list,
        select_fields: list,
        num_documents: int = 5,
    ) -> int:
        """
        Calculates the average document size, averaged over num_documents.
        """
        documents = self.datasets.documents.get_where(
            dataset_id,
            filters=filters,
            select_fields=select_fields,
            page_size=num_documents,
        )["documents"]
        return (
            sum(map(lambda document: len(orjson.dumps(document)), documents))
            // num_documents
        )

    def _determine_optimal_chunk_size(
        self, dataset_id: str, filters: list, select_fields: list
    ) -> int:
        """
        Determines the "optimal" chunk size in terms of bytes. The optimal
        chunk size is calculated by divinding 50 MB by the average document
        size. 50 MB is a chosen constant, which is a nice size for the backend
        to handle.
        """
        document_size = self._get_avg_document_size(dataset_id, filters, select_fields)
        return (50 * MB_TO_BYTE) // document_size

    def _set_chunk_size(
        self,
        dataset_id: str,
        filters: list,
        select_fields: list,
        retrieve_chunk_size: Optional[int],
    ) -> int:
        """
        Sets the chunk size to an "optimal" size is the chunk size is not
        specified. Else, makes sure thathe chunk size is within tolerance.
        """
        if retrieve_chunk_size is None:
            chunk_size = self._determine_optimal_chunk_size(
                dataset_id, filters, select_fields
            )
            print(f"Chunk size set to {chunk_size}")
            self.logger.info(f"Chunk size set to {chunk_size}")
            retrieve_chunk_size = chunk_size

        if retrieve_chunk_size > 10_000:
            # The backend API has an upper bound of 10,000 on the number of
            # documents that are able to be pulled at once.
            chunk_size = 10_000
            self.logger.info(f"Chunk size set to {chunk_size}")
            return chunk_size
        else:
            return retrieve_chunk_size

    def _get_cache_path(self, dataset_id, select_fields) -> Path:
        """
        Gets (and creates) the cache path. The path is denoted by the hash of
        the fields as a simple means to distinguish different cache recalls
        of the same dataset.
        """
        cache_path = (
            Path(user_cache_dir())
            / Path("relevanceai")
            / Path(f"{dataset_id}-{hash(frozenset(select_fields))}")
        )
        if not cache_path.exists():
            cache_path.mkdir(parents=True)
            self.logger.info(f"Created directory {cache_path}")

        return cache_path

    def _create_subset_function(
        self,
        dataset_id: str,
        update_function: Callable[..., list],
        updating_args: dict,
        updated_dataset_id: Optional[str],
        filters: Optional[list],
        select_fields: Optional[list],
        use_json_encoder: bool,
        include_vector: bool,
        insert: bool,
        use_cache: bool,
    ):
        """
        A helper function to create the task coroutine.
        """
        cache_path = self._get_cache_path(dataset_id, select_fields)

        async def pull_update_push_subset(
            num: int, page_size: int, cursor: Optional[str]
        ):
            save_file = cache_path / f"pull-{num}.json"
            if save_file.exists() and use_cache:
                with open(save_file, "rb") as infile:
                    documents = orjson.loads(infile.read())
            else:
                response = await self.datasets.documents.get_where_async(
                    dataset_id,
                    filters=filters,
                    cursor=cursor,
                    page_size=page_size,
                    select_fields=select_fields,
                    include_vector=include_vector,
                )

                documents = response["documents"]

                with open(save_file, "wb") as outfile:
                    outfile.write(orjson.dumps(documents))

            try:
                updated_documents = update_function(documents, **updating_args)
            except Exception as e:
                self.logger.error("Your updating function does not work: " + str(e))
                traceback.print_exc()
                return

            updated_ids = [document["_id"] for document in documents]

            inserted = await self._process_documents(
                dataset_id=updated_dataset_id
                if updated_dataset_id is not None
                else dataset_id,
                documents=updated_documents,
                insert=insert,
                use_json_encoder=use_json_encoder,
            )

            return inserted, updated_ids

        return pull_update_push_subset

    def _validate_cache(
        self, use_cache: bool, dataset_id: str, select_fields: list
    ) -> bool:
        """
        A means to validate the cache by determing that the user desires to
        access the cache and the cache is non-empty.
        """
        num_cached = len(
            list(self._get_cache_path(dataset_id, select_fields).iterdir())
        )
        return use_cache and bool(num_cached)

    def _get_cursor(
        self, dataset_id: str, filters: list, select_fields: list, chunk_size: int
    ) -> str:
        """
        Retrieves the cursor from the backend.
        """
        # The cursor is constrained by its first query. Specifically,
        # if the query from which we get the cursor has a certain page
        # size, further accesses through the cursor will be the same
        # page size. Therefore, in the tasks below, the first task
        # does a proper query of the first {retrieve_chunk_size}
        # documents. Subsequent calls use the cursor to access the
        # next {retrieve_chunk_size} documents {num_requests-1} times.
        # The 0 is there to make the aforementioned statement
        # explicit.
        return self.datasets.documents.get_where(
            dataset_id,
            filters=filters,
            select_fields=select_fields,
            page_size=chunk_size,
            include_vector=False,  # Set to false to for speed
        )["cursor"]

    def _get_task_parameters(
        self,
        valid_cache: bool,
        chunk_size: int,
        dataset_id: str,
        filters: list,
        select_fields: list,
    ) -> Tuple[int, Optional[str]]:
        """
        Retrieves the number of requests (num_requests) and the cursor.
        """
        # num_requests will determine how many requests are sent.
        # Example: Suppose the number of documents is 1254 and
        # retrieve_chunk_size is 100. Then the number of requests would
        # be 1254 // 100, which would be 12, and one, which amounts to
        # 13 requests. This must be defined ahead of time because the
        # cursor predetermines the number of documents to retrieve on
        # the first call. So, The first 12 calls would each get 100
        # documents and call 13 would retrieve the remaining 54.
        if valid_cache:
            print("Accessing cache, disregarding chunk size")
            num_requests = len(
                list(self._get_cache_path(dataset_id, select_fields).iterdir())
            )
            cursor = None
        else:
            # If use_cache is False or num_cached is zero cache will not
            # be accessed.
            num_documents = self.get_number_of_documents(dataset_id, filters)
            num_requests = (num_documents // chunk_size) + 1
            cursor = self._get_cursor(dataset_id, filters, select_fields, chunk_size)

        return num_requests, cursor

    def _get_tasks(
        self,
        valid_cache: bool,
        chunk_size: int,
        dataset_id: str,
        filters: list,
        select_fields: list,
        subset_function,
    ) -> list:
        """
        Gets the list of tasks to be executed in the event loop.
        """
        num_requests, cursor = self._get_task_parameters(
            valid_cache, chunk_size, dataset_id, filters, select_fields
        )
        if valid_cache:
            return [
                subset_function(task_num, 0, cursor) for task_num in range(num_requests)
            ]
        else:
            return [
                subset_function(task_num, chunk_size, None)
                if not task_num
                else subset_function(task_num, 0, cursor)
                for task_num in range(num_requests)
            ]

    def _get_log_file(self, dataset_id) -> str:
        """
        Creates a log file name.
        """
        log_file = "_".join(
            [
                dataset_id,
                str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
                "pull_update_push.log",
            ]
        )
        self.logger.info(f"Created {log_file}")

        return log_file

    def _execute_event_loop(
        self,
        dataset_id: str,
        update_function: Callable[..., list],
        updating_args: dict,
        updated_dataset_id: Optional[str],
        retrieve_chunk_size: Optional[int],
        filters: list,
        select_fields: list,
        use_json_encoder: bool,
        include_vector: bool,
        show_progress_bar: bool,
        insert: bool,
        use_cache: bool,
    ) -> list:
        """
        Primary driver of pull_update_push_async. Retrieves the relevant
        items then executes them in the event loop.
        """
        pull_update_push_subset = self._create_subset_function(
            dataset_id,
            update_function,
            updating_args,
            updated_dataset_id,
            filters,
            select_fields,
            use_json_encoder,
            include_vector,
            insert,
            use_cache,
        )
        valid_cache = self._validate_cache(use_cache, dataset_id, select_fields)
        chunk_size = self._set_chunk_size(
            dataset_id, filters, select_fields, retrieve_chunk_size
        )
        tasks = self._get_tasks(
            valid_cache,
            chunk_size,
            dataset_id,
            filters,
            select_fields,
            pull_update_push_subset,
        )

        threaded_loop = EventLoop(tasks, show_progress_bar)
        threaded_loop.start()
        threaded_loop.execute_tasks()
        threaded_loop.terminate()

        failed_documents = []
        for future in threaded_loop.futures:
            inserted, _ = future.result()
            failed_documents.extend(inserted["failed_documents"])

        return failed_documents


class BatchInsertAsyncClient(BatchInsertAsyncHelpers):
    def pull_update_push_async(
        self,
        dataset_id: str,
        update_function: Callable,
        updating_args: Optional[dict] = None,
        updated_dataset_id: Optional[str] = None,
        log_file: str = None,
        retrieve_chunk_size: Optional[int] = None,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        use_json_encoder: bool = True,
        include_vector: bool = True,
        show_progress_bar: bool = True,
        insert: bool = False,
        use_cache: bool = False,
        log_to_file: bool = True,
    ) -> dict:
        """
        Loops through every document in your collection and applies a function (that is specified by you) to the documents.
        These documents are then uploaded into either an updated collection, or back into the original collection.

        Parameters
        ----------
        dataset_id: str
            The dataset_id of the collection where your original documents are

        update_function: Callable
            A function created by you that converts documents in your original
            collection into the updated documents. The function must contain a
            field which takes in a list of documents from the original
            collection. The output of the function must be a list of updated
            documents.

        updating_args: dict
            Additional arguments to your update_function, if they exist. They
            must be in the format of {'Argument': Value}

        updated_dataset_id: str
            The dataset_id of the collection where your updated documents are
            uploaded into. If 'None', then your original collection will be
            updated.

        log_file: str
            The log file to direct any information or issues that may crop up.
            If no log file is specified, one will automatically be created.

        retrieve_chunk_size: int
            The number of documents that are received from the original
            collection with each loop iteration.

        filters: list
            A list of filters to apply on the retrieval query.

        select_fields: list
            A list of fields to query over.

        use_json_encoder : bool
            If True, automatically convert documents to json encodable format.

        include_vector: bool
            If True, includes vectors in the updating query.

        show_progress_bar: bool
            If True, shows a progress bar.

        insert: bool
            If True, inserts rather than updates an already-existing dataset.

        use_cache: bool
            If True, and if the cache is not empty, uses cached documents.

        log_to_file: bool
            If True, logs errors to a file.
        """
        updating_args = {} if updating_args is None else updating_args
        filters = [] if filters is None else filters
        select_fields = [] if select_fields is None else select_fields

        if not callable(update_function):
            raise TypeError(
                "Your update function needs to be a function! "
                + "Please read the documentation if it is not."
            )

        if not (retrieve_chunk_size is None or isinstance(retrieve_chunk_size, int)):
            raise TypeError(f"{retrieve_chunk_size} must be None or an integer.")

        if log_file is None:
            log_file = self._get_log_file(dataset_id)
        with FileLogger(fn=log_file, verbose=True, log_to_file=log_to_file):
            failed_documents = self._execute_event_loop(
                dataset_id,
                update_function,
                updating_args,
                updated_dataset_id,
                retrieve_chunk_size,
                filters,
                select_fields,
                use_json_encoder,
                include_vector,
                show_progress_bar,
                insert,
                use_cache,
            )
            if failed_documents:
                # This will be picked up by FileLogger
                print("The following documents failed to be updated/inserted:")
                for failed_document in failed_documents:
                    print(f"  * {failed_document}")

        self.logger.success("Pull, update, and push is complete!")

        return {"failed_documents": failed_documents}
