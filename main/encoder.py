from env import mongo_client, db
from helper_functions import timer, debug
import pandas 
import pickle

@debug
@timer
def encode(mongo_client, db):
    db.user_follower.update_many({'expired': {'$exists': False}}, {'$set': {'expired': '0'}})
    try: 
        with open('./main/counts', 'rb') as f:
            counts = pickle.load(f)
    except FileNotFoundError:
        counts = {}
    docs = db.user_follower.find({'expired': '0'}, {'user_id': 1, 'follower_user_id': '1'})
    for doc in docs: 
        if doc['user_id'] in counts:
            counts[doc['user_id']] += 1
        else:
            counts[doc['user_id']] = 1
        if doc['follower_user_id'] in counts:
            counts[doc['follower_user_id']] += 1
        else:
            counts[doc['follower_user_id']] = 1
        db.user_follower.update_one({'_id': doc['_id']}, {'$set': {'expired': '1'}})
    with open('./main/counts', 'wb') as file:
        pickle.dump(counts, file)
if __name__ == "__main__":
    encode(mongo_client, db)
