import logging


def get_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter(
        fmt='%(levelname)s : %(process)d : %(asctime)s : %(name)s \t| %(message)s',
        datefmt='%H:%M'))
    handler2 = logging.FileHandler(filename=filename)
    handler2.setFormatter(logging.Formatter(
        fmt='%(levelname)s : %(asctime)s : %(name)s \t| %(message)s',
        datefmt='%m-%d %H:%M'))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
