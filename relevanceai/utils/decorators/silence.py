import os
import sys


def block_print():
    sys.stdout = open(os.devnull, "w")


def enable_print():
    sys.stdout = sys.__stdout__


def silence(func):
    def wrapper(*args, **kwargs):
        block_print()
        func(*args, **kwargs)
        enable_print()

    return wrapper
