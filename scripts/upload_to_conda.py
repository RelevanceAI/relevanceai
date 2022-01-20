"""
script to upload to conda :)
Should be run from root of directory
Requirements: 
- Install anaconda
- `conda install conda==4.10.3` (newer versions are broken)
- Run `conda install anaconda`
- `anaconda login`
- `conda build . -u relevance`
Then from the output, use this:
- conda convert /Users/jacky.wong/opt/anaconda3/conda-bld/osx-64/relevanceai-0.28.0-py39_0.tar.bz2 -p all -o channel
- conda config --add channels relevance
- conda config --add channels conda-forge
"""

import os 
from pathlib import Path
from tqdm.auto import tqdm

# os.system("conda build .")
# TODO:
# anaconda upload \
#     /Users/jacky.wong/opt/anaconda3/conda-bld/osx-64/relevanceai-0.28.0-0.tar.bz2
# anaconda_upload is not set.  Not uploading wheels: []
# Automate the above

import subprocess
proc = subprocess.Popen(["conda", "build ."], stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()


# from observing the outputs, these seem to be the main file formats

files_to_upload = list(Path("channel").rglob("*.whl")) +  list(Path("channel").rglob("*tar.gz")) + list(Path("channel").rglob("*.bz2"))
for fn in tqdm(files_to_upload):
    print(fn)
    cmd = f"anaconda upload --force -u relevance {fn}"
    print(cmd)
    os.system(cmd)
