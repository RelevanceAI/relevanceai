"""Local logger for pull_update_push.
"""
import os
from os import PathLike
from typing import Union
from relevanceai.config import CONFIG
from relevanceai.logger import LoguruLogger


class PullUpdatePushLocalLogger(LoguruLogger):
    """This logger class is specifically for pull_update_push to log
    failures locally as opposed to on the cloud.
    """

    def __init__(self, filename: Union[str, bytes]):
        """Filename for failed IDs"""
        self.filename = filename
        self.config = CONFIG
        super().__init__()

    def log_ids(self, id_list, verbose: bool = True):
        """Log the failed IDs to the file"""
        with open(self.filename, "a") as f:
            for _id in id_list:
                f.write(_id + "\n")
        if verbose:
            self.logger.info("Logging")

    def count_ids_in_fn(self) -> int:
        """Returns total count of failed IDs"""
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                lines = f.readlines()
            return len(set(lines))
        return 0
