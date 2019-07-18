import logging
from pymongo import MongoClient

mongo_client = MongoClient("127.0.0.1:27017")
db = mongo_client.rumi