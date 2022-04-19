"""
Run this file if you want to generate coverage.cfg and you want to exclude many file/dir paths
"""

RUN_EXCLUDE = [
    "*__init__.py",
    "relevanceai/_api/*",
    "relevanceai/dashboard/*",
    "relevanceai/operations/auto/*",
    "relevanceai/operations/cluster/base.py",
    "relevanceai/operations/cluster/centroids.py",
    "relevanceai/operations/cluster/groupby.py",
    "relevanceai/operations/cluster/ops.py",
    "relevanceai/operations/cluster/partial.py",
    "relevanceai/operations/cluster/sub.py",
    "relevanceai/operations/cluster/utils.py",
    "relevanceai/operations/labels/*",
    "relevanceai/operations/preprocessing/*",
    "relevanceai/operations/unstruc/*",
    "relevanceai/operations/vector/*",
    "relevanceai/operations/visualise/*",
    "relevanceai/reports/*",
    "relevanceai/workflow/*",
]
REPORT_EXCLUDE = RUN_EXCLUDE

COVERAGE_CFG_PATH = "coverage/coverage.cfg"


def main():
    lines = []

    lines.append("[run]")
    lines.append("omit = " + ",".join(RUN_EXCLUDE))
    lines.append("")
    lines.append("[report]")
    lines.append("omit = " + ",".join(REPORT_EXCLUDE))

    text = "\n".join(lines)

    with open(COVERAGE_CFG_PATH, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
