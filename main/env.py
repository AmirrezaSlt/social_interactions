import logging
from pymongo import MongoClient

mongo_client = MongoClient("127.0.0.1:27017")
db = mongo_client.testcn10


def create_logger():
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(r"./logs/logfile.log")
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


logger = create_logger()
