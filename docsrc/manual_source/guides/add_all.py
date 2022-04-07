import os

GUIDES = "docsrc/manual_source/guides"

top = """Guides
------------


.. toctree::
    :maxdepth: 4

"""

guides = [
    "\t" + file.split(".rst")[0] + "\n"
    for file in os.listdir(GUIDES)
    if file.endswith(".rst") and "guide" in file
]

with open(GUIDES + "/index.rst", "w", encoding="utf-8") as f:
    f.write(top)
    for guide in guides:
        f.write(guide)
