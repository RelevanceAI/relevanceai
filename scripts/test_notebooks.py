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
parser.add_argument('path')
args = parser.parse_args()
###############################################################################
# Helper Functions
###############################################################################

def check_latest_version(name):
    latest_version = str(subprocess.run([
        sys.executable, '-m', 'pip', 'install', '{}==random'.format(name)], 
        capture_output=True, text=True))
    latest_version = latest_version[latest_version.find('(from versions:')+15:]
    latest_version = latest_version[:latest_version.find(')')]
    latest_version = latest_version.replace(' ','').split(',')[-1]

    current_version = str(subprocess.run([
        sys.executable, '-m', 'pip', 'show', '{}'.format(name)], 
        capture_output=True, text=True))
    current_version = current_version[current_version.find('Version:')+8:]
    current_version = current_version[:current_version.find('\\n')].replace(' ','') 

    if latest_version == current_version:
        return True
    else:
        return False

def get_latest_version(name):
    latest_version = str(subprocess.run([sys.executable, '-m', 'pip', 'install', '{}==random'.format(name)], capture_output=True, text=True))
    latest_version = latest_version[latest_version.find('(from versions:')+15:]
    latest_version = latest_version[:latest_version.find(')')]
    latest_version = latest_version.replace(' ','').split(',')[-1]
    return latest_version


###############################################################################
# Update SDK version and test
###############################################################################


DOCS_PATH = Path(args.path) / 'docs'
RELEVANCEAI_SDK_VERSION_LATEST = get_latest_version('RelevanceAI')
# RELEVANCEAI_SDK_VERSION_LATEST = 'latest'
PIP_INSTALL_REGEX = f'"!pip install .* RelevanceAI==.*"'
PIP_INSTALL_LATEST = f'"!pip install -U RelevanceAI=={RELEVANCEAI_SDK_VERSION_LATEST}"'

def notebook_find_replace(notebook, find_str_regex, replace_str):

    with open(notebook, 'r') as f:
        lines = f.readlines()

    with open(notebook, 'w') as f:
        for line in lines:          
            if bool(re.search(find_str_regex, line)):
                find_str = re.search(find_str_regex, line).group()
                
                # if find_str == replace_str: continue
                print(f'Find: \n{find_str_regex}')
                print(f'Replace: \n{find_str}\n{replace_str}\n')
                line = line.replace(find_str, replace_str)
                
            f.write(line)

# notebook = Path.cwd() / 'examples' / 'Intro_to_Relevance_AI.ipynb'

for notebook in DOCS_PATH.rglob('*.ipynb'):
    print(notebook)

    ## Update to latest version
    notebook_find_replace(notebook, PIP_INSTALL_REGEX, PIP_INSTALL_LATEST)

    ## Replace Client with test creds
    CLIENT_INSTANTIATION_REGEX = '"client.*Client(.*)"'
    TEST_PROJECT = os.getenv('TEST_PROJECT')
    TEST_API_KEY = os.getenv('TEST_API_KEY')
    CLIENT_INSTANTIATION_TEST = f'"client = Client(project=\\"{TEST_PROJECT}\\", api_key=\\"{TEST_API_KEY}\\")"'
    CLIENT_INSTANTIATION_BASE = f'"client = Client()"'

    ## Temporarily updating notebook with test creds
    notebook_find_replace(notebook, CLIENT_INSTANTIATION_REGEX, CLIENT_INSTANTIATION_TEST)

    ## Execute notebook with test creds
    with open(notebook, 'r') as f:
        ## Execute notebook
        print(f'Executing notebook: \n{notebook} with SDK version {RELEVANCEAI_SDK_VERSION_LATEST}')
        nb_in = nbformat.read(f, nbformat.NO_CONVERT)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        nb_out = ep.preprocess(nb_in)

    ## Replace creds with previous 
    notebook_find_replace(notebook, CLIENT_INSTANTIATION_REGEX, CLIENT_INSTANTIATION_BASE)
