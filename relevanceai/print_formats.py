"""Utility functions for useful printing formats.
"""


def print_hashtag_message(message) -> None:
    loud_format = "###" + len(message) * "#" + "###"
    message_format = "## " + message + " ##"

    print(loud_format)
    print(message_format)
    print(loud_format)


class COLOR:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_bold_message(message):
    print(COLOR.BOLD + message + COLOR.END)
