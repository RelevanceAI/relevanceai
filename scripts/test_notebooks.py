#!/usr/bin/env python3

import os
from pathlib import Path
import re
import subprocess
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

###############################################################################
# Helper Functions
###############################################################################


def get_latest_version(name):
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


def check_latest_version(name):
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


###############################################################################
# Update SDK version and test
###############################################################################

DOCS_PATH = Path(args.path) / "docs"
RELEVANCEAI_SDK_VERSION_LATEST = get_latest_version("RelevanceAI")
# RELEVANCEAI_SDK_VERSION_LATEST = 'latest'
PIP_INSTALL_SENT_REGEX = f'".*pip install .* RelevanceAI.*==.*"'
PIP_INSTALL_STR_REGEX = f"==.*[0-9]"
PIP_INSTALL_STR_REPLACE = f"=={RELEVANCEAI_SDK_VERSION_LATEST}"


def notebook_find_replace(notebook, find_sent_regex, find_str_regex, replace_str):

    with open(notebook, "r") as f:
        lines = f.readlines()

    with open(notebook, "w") as f:
        for i, line in enumerate(lines):
            if bool(re.search(find_sent_regex, line)):
                find_sent = re.search(find_sent_regex, line)
                if find_sent:
                    find_sent = find_sent.group()
                    print(f"Found: {find_sent}\n")

                    # if find_str == replace_str: continue
                    print(f"Find string: {find_str_regex}")
                    find_replace_str = re.search(find_str_regex, find_sent)
                    if find_replace_str:
                        find_replace_str = find_replace_str.group()
                        print(f"Found: {find_replace_str}\n")

                        print(f"Replace: \n{find_replace_str}\n{replace_str}\n")
                        line = line.replace(find_replace_str, replace_str)

                        print(f"Updated:")
                        print(line.strip())
                    else:
                        print(f"Not found: {find_replace_str}\n")
                else:
                    print(f"Not found: {find_sent_regex}\n")

            f.write(line)


## Env vars
CLIENT_INSTANTIATION_SENT_REGEX = '"client.*Client(.*)"'
TEST_PROJECT = os.getenv("TEST_PROJECT")
TEST_API_KEY = os.getenv("TEST_API_KEY")
CLIENT_INSTANTIATION_STR_REGEX = "\((.*?)\)"
CLIENT_INSTANTIATION_STR_REPLACE = (
    f'(project=\\"{TEST_PROJECT}\\", api_key=\\"{TEST_API_KEY}\\")'
)

CLIENT_INSTANTIATION_BASE = f'"client = Client()"'


for notebook in Path(DOCS_PATH).glob("**/*.ipynb"):
    print(notebook)

    ## Update to latest version
    notebook_find_replace(
        notebook, PIP_INSTALL_SENT_REGEX, PIP_INSTALL_STR_REGEX, PIP_INSTALL_STR_REPLACE
    )

    ## Temporarily updating notebook with test creds
    notebook_find_replace(
        notebook,
        CLIENT_INSTANTIATION_SENT_REGEX,
        CLIENT_INSTANTIATION_STR_REGEX,
        CLIENT_INSTANTIATION_STR_REPLACE,
    )

    ## Execute notebook with test creds
    with open(notebook, "r") as f:
        print(
            f"Executing notebook: \n{notebook} with SDK version {RELEVANCEAI_SDK_VERSION_LATEST}"
        )
        nb_in = nbformat.read(f, nbformat.NO_CONVERT)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        nb_out = ep.preprocess(nb_in)

    ## Replace creds with previous
    notebook_find_replace(
        notebook,
        CLIENT_INSTANTIATION_SENT_REGEX,
        CLIENT_INSTANTIATION_STR_REGEX,
        CLIENT_INSTANTIATION_BASE,
    )
