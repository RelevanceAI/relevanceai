#!/usr/bin/env python3

from typing import Dict
import os
from pathlib import Path
import re
import subprocess
import sys
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import traceback

from relevanceai.concurrency import multiprocess

import logging
import argparse


###############################################################################
# Helper Functions
###############################################################################

README_NOTEBOOK_ERROR_FPATH = "readme_notebook_error_log.txt"


def get_latest_version(name: str):
    latest_version = str(
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "{}==random".format(name)],
            capture_output=True,
            text=True,
        )
    )
    latest_version = latest_version[latest_version.find("(from versions:") + 15 :]
    latest_version = latest_version[: latest_version.find(")")]
    latest_version = latest_version.replace(" ", "").split(",")[-1]
    return latest_version


def check_latest_version(name: str):
    latest_version = get_latest_version(name)
    current_version = str(
        subprocess.run(
            [sys.executable, "-m", "pip", "show", "{}".format(name)],
            capture_output=True,
            text=True,
        )
    )
    current_version = current_version[current_version.find("Version:") + 8 :]
    current_version = current_version[: current_version.find("\\n")].replace(" ", "")

    if latest_version == current_version:
        return True
    else:
        return False


def notebook_find_replace(
    fname: str, find_sent_regex: str, find_str_regex: str, replace_str: str
):
    logging.info(f"\tInput: {fname}")
    notebook_json = json.loads(open(fname).read())

    for cell in notebook_json["cells"]:
        if bool(re.search(find_sent_regex, str(cell["source"]))):
            logging.debug(f"Found sentence: {str(cell['source'])}")
            logging.debug(f"Find string regex: {find_str_regex}")
            for i, cell_source in enumerate(cell["source"]):
                if bool(re.search(find_str_regex, cell_source)):
                    find_replace_str = re.search(find_str_regex, cell_source).group()
                    logging.debug(
                        f"Found str within sentence: {find_replace_str.strip()}"
                    )
                    logging.debug(f"Replace str: {replace_str}")
                    cell_source = cell_source.replace(find_replace_str, replace_str)
                    logging.debug(f"Updated: {cell_source.strip()}")
                    cell["source"][i] = cell_source

    logging.info(f"\tOutput file: {fname}")
    json.dump(notebook_json, fp=open(fname, "w"), indent=4)


def update_pip_for_shell(fname: str, shell: str = "zsh"):
    logging.info(f"\tInput: {fname}")
    notebook_json = json.loads(open(fname).read())

    PIP_INSTALL_SENT_REGEX = f".*pip install.*"
    for cell in notebook_json["cells"]:
        if bool(re.search(PIP_INSTALL_SENT_REGEX, str(cell["source"]))):
            logging.debug(f"Source:{str(cell['source'])}")
            for i, cell_source in enumerate(cell["source"]):
                ## zsh uses square brackets for globbing eg. pip install -U 'RelevanceAI[notebook]==0.33.2'
                for m in re.finditer(PIP_INSTALL_SENT_REGEX, cell_source):
                    pip_install_match = m.group()
                    pip_install_str = pip_install_match.split()[:-1]
                    package_install_str = pip_install_match.split()[-1]
                    if shell == "zsh":
                        new_package_install_str = f"'{package_install_str}'"
                        logging.debug(f"\tUpdating for zsh {new_package_install_str}")
                    elif shell == "bash":
                        new_package_install_str = package_install_str.replace(
                            "'", ""
                        ).replace('"', "")
                        logging.debug(f"\tUpdating for bash {new_package_install_str}")
                    cell_source = cell_source.replace(
                        package_install_str, new_package_install_str
                    )

                logging.debug(cell["source"][i])
                cell["source"][i] = cell_source

    logging.info(f"\tOutput file: {fname}")
    json.dump(notebook_json, fp=open(fname, "w"), indent=4)


###############################################################################
# Update SDK version and test
###############################################################################


def execute_notebook(notebook: str, notebook_args: Dict):
    try:
        print(notebook)

        if notebook_args["multiprocess"]:
            if isinstance(notebook, list):
                notebook = notebook[0]

        if os.getenv("SHELL") and "zsh" in os.getenv("SHELL"):
            update_pip_for_shell(notebook, shell="zsh")

        ## Update to latest version
        notebook_find_replace(notebook, **notebook_args["pip_install_version_args"])

        ## Temporarily updating notebook with test creds
        notebook_find_replace(notebook, **notebook_args["client_instantiation_args"])

        ## Execute notebook with test creds
        with open(notebook, "r") as f:
            print(
                f"\nExecuting notebook: \n{notebook} with SDK version {notebook_args['relevanceai_sdk_version']}"
            )
            nb_in = nbformat.read(f, nbformat.NO_CONVERT)
            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
            nb_out = ep.preprocess(nb_in)

        ## Replace with bash
        if os.getenv("SHELL") and "zsh" in os.getenv("SHELL"):
            update_pip_for_shell(notebook, shell="bash")

        # Replace client test creds
        notebook_find_replace(
            notebook, **notebook_args["client_instantiation_base_args"]
        )
        return
    except Exception as e:

        ## Replace with bash
        if os.getenv("SHELL") and "zsh" in os.getenv("SHELL"):
            update_pip_for_shell(notebook, shell="bash")

        ## Replace client test creds
        notebook_find_replace(
            notebook, **notebook_args["client_instantiation_base_args"]
        )

        exception_reason = traceback.format_exc()
        ERROR_MESSAGE = f"{notebook}\n{exception_reason}\n"
        print(
            f"{ERROR_MESSAGE}\n==============",
            file=open(README_NOTEBOOK_ERROR_FPATH, "a"),
        )
        if notebook_args["multiprocess"]:
            return {
                "notebook": notebook.__str__(),
                "Exception reason": exception_reason,
            }
        raise ValueError(f"{ERROR_MESSAGE}")


###############################################################################
# Main
###############################################################################


def main(args):
    logging_level = logging.DEBUG if args.debug else logging.INFO
    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging_level)
    logging.basicConfig(level=logging_level)

    DOCS_PATH = Path(args.path) / "docs"

    NOTEBOOK_IGNORE = (
        open(Path(__file__).parent / "notebook_ignore.txt").read().strip().splitlines()
    )
    print(f"NOTEBOOK_IGNORE: {NOTEBOOK_IGNORE}")
    RELEVANCEAI_SDK_VERSION = (
        args.version if args.version else get_latest_version(args.package_name)
    )
    logging.info(
        f"Executing notebook test with {args.package_name}=={RELEVANCEAI_SDK_VERSION}\n\n"
    )

    PIP_INSTALL_SENT_REGEX = f".*pip install .* {args.package_name}.*==.*"
    PIP_INSTALL_VERSION_STR_REGEX = f"==.*[0-9]"
    PIP_INSTALL_VERSION_STR_REPLACE = f"=={RELEVANCEAI_SDK_VERSION}"
    pip_install_version_args = {
        "find_sent_regex": PIP_INSTALL_SENT_REGEX,
        "find_str_regex": PIP_INSTALL_VERSION_STR_REGEX,
        "replace_str": PIP_INSTALL_VERSION_STR_REPLACE,
    }

    ## Env vars
    CLIENT_INSTANTIATION_SENT_REGEX = "client.*Client(.*)"
    CLIENT_INSTANTIATION_STR_REGEX = "\((.*?)\)"

    TEST_PROJECT = os.getenv("TEST_PROJECT")
    TEST_API_KEY = os.getenv("TEST_API_KEY")
    TEST_ACTIVATION_TOKEN = os.getenv("TEST_ACTIVATION_TOKEN")
    if TEST_ACTIVATION_TOKEN:
        CLIENT_INSTANTIATION_STR_REPLACE = f'(token="{TEST_ACTIVATION_TOKEN}")'
    elif TEST_PROJECT and TEST_API_KEY:
        CLIENT_INSTANTIATION_STR_REPLACE = (
            f'(project=\\"{TEST_PROJECT}\\", api_key=\\"{TEST_API_KEY}\\")'
        )
    else:
        raise ValueError(
            f"Please set the client test credentials\n\
            export TEST_ACTIVATION_TOKEN=xx or\nexport TEST_PROJECT=xx\nexport TEST_API_KEY=xx"
        )

    CLIENT_INSTANTIATION_BASE = f"client = Client()"
    client_instantiation_args = {
        "find_sent_regex": CLIENT_INSTANTIATION_SENT_REGEX,
        "find_str_regex": CLIENT_INSTANTIATION_STR_REGEX,
        "replace_str": CLIENT_INSTANTIATION_STR_REPLACE,
    }

    client_instantiation_base_args = {
        "find_sent_regex": CLIENT_INSTANTIATION_SENT_REGEX,
        "find_str_regex": CLIENT_INSTANTIATION_SENT_REGEX,
        "replace_str": CLIENT_INSTANTIATION_BASE,
    }

    if args.notebooks:
        notebooks = args.notebooks
        if len(notebooks) == 1:
            if Path(notebooks[0]).is_dir():
                notebooks = [f for f in Path(notebooks[0]).glob("**/*.ipynb")]
    else:
        ## All notebooks
        notebooks = [
            x[0] if isinstance(x, list) else x
            for x in list(Path(DOCS_PATH).glob("**/*.ipynb"))
        ]

    ## Filter checkpoints
    notebooks = [f for f in notebooks if ".ipynb_checkpoints" not in str(f)]

    ## Filter notebooks
    if args.notebook_ignore:
        notebooks = [n for n in notebooks if n not in NOTEBOOK_IGNORE]

    if not notebooks:
        print(f"No notebooks found not in {NOTEBOOK_IGNORE}")
        sys.exit(1)

    static_args = {
        "relevanceai_sdk_version": RELEVANCEAI_SDK_VERSION,
        "pip_install_version_args": pip_install_version_args,
        "client_instantiation_args": client_instantiation_args,
        "client_instantiation_base_args": client_instantiation_base_args,
        "multiprocess": False if args.no_multiprocess else True,
    }

    with open(README_NOTEBOOK_ERROR_FPATH, "w") as f:
        f.write("")

    if args.no_multiprocess:
        logging.info("Executing sequentially")
        for notebook in notebooks:
            execute_notebook(notebook, static_args)
    else:
        logging.info("Executing in multiprocess mode")
        results = multiprocess(
            func=execute_notebook,
            iterables=notebooks,
            static_args=static_args,
            chunksize=1,
        )

        # results = [execute_notebook(n) for n in ALL_NOTEBOOKS]
        results = [r for r in results if r is not None]
        if len(results) > 0:
            for r in results:
                print(r.get("Exception reason"))
                print("============")
                print(r.get("notebook"))
            raise ValueError(f"You have errored notebooks {results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    PACKAGE_NAME = "RelevanceAI"
    ROOT_PATH = Path(__file__).parent.resolve() / ".."

    # try:
    #     README_VERSION = open(ROOT_PATH / "__version__").read()
    # except FileNotFoundError as e:
    #     print(f"File not found: {e}")
    #     print(f"Loading file from latest Pip package release")

    parser.add_argument("-d", "--debug", action="store_true", help="Run debug mode")
    parser.add_argument("-p", "--path", default=ROOT_PATH, help="Path of root folder")
    parser.add_argument(
        "-pn", "--package-name", default=PACKAGE_NAME, help="Package Name"
    )
    parser.add_argument("-v", "--version", default=None, help="Package Version")
    parser.add_argument(
        "-n",
        "--notebooks",
        nargs="+",
        default=None,
        help="List of notebooks to execute",
    )
    parser.add_argument(
        "-nm",
        "--no-multiprocess",
        action="store_true",
        help="Whether to run multiprocessing",
    )
    parser.add_argument(
        "-i",
        "--notebook-ignore",
        action="store_true",
        help="Whether to include notebook ignore list",
    )
    args = parser.parse_args()

    main(args)
