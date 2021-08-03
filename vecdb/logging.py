import logging
import time

def create_logger(orig_func, log_file, log_console):
    logger = logging.getLogger(orig_func.__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    if (logger.hasHandlers()):
        logger.handlers.clear()

    if log_file == True:
        file_handler = logging.FileHandler('vecdb.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_console == True:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def logger(orig_func):
    def wrapper(*args, **kwargs):

        if args[0].config.logging is True:

            logger = create_logger(orig_func, args[0].config.log_to_file, args[0].config.log_to_console)

            t1 = time.time()
            results = orig_func(*args, **kwargs)
            t2 = time.time() - t1
            logger.info(f'Ran in {t2} seconds with args {args} and kwargs {kwargs}')

            return results

        else:
            return orig_func(*args, **kwargs)
    
    return wrapper