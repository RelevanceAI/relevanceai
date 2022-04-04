"""
A script for generating .rst files for /docsrc from .ipynb files in this folder
"""
import os
import subprocess

BASE_ARGS = [
    "jupyter",
    "nbconvert",
    "--to",
    "rst",
]

DOCSRC_GUIDES = "docsrc/manual_source/guides"


def main():
    guide_rst = os.listdir(DOCSRC_GUIDES)

    for file in guide_rst:
        if file.endswith(".rst") and "guide" in file:
            args = ["rm", "-rf", f"{DOCSRC_GUIDES}/{file}"]
            subprocess.run(args)

    guides = os.listdir("guides")

    for file in guides:
        if file.endswith(".ipynb"):
            file, ext = file.split(".")

            args = BASE_ARGS + [f"guides/{file}.ipynb"]
            subprocess.run(args)

            move = ["mv", f"guides/{file}.rst", DOCSRC_GUIDES]
            subprocess.run(move)


if __name__ == "__main__":
    main()
