import collections as cnt
from env import mongo_client, db
from helper_functions import timer, debug
import pandas 
import pickle

@debug
@timer
def counter(mongo_client, db):
    db.user_follower.update_many({'read': {'$exists': False}}, {'$set': {'read': '0'}})
    try: 
        with open('./main/counts', 'rb') as f:
            counts = pickle.load(f)
    except FileNotFoundError:
        return reset_counter(mongo_client, db)
    docs = db.user_follower.find({'read': '0'}, {'user_id': 1, 'follower_user_id': 1})
    for doc in docs: 
        counts[doc['user_id']] += 1
        counts[doc['follower_user_id']] += 1
        db.user_follower.update_one({'_id': doc['_id']}, {'$set': {'read': '1', 'expired': '0'}})
    with open('./main/counts', 'wb') as file:
        pickle.dump(counts, file)
    return integer_encoding(mongo_client, db, counts)
    
def reset_counter(mongo_client, db): 
    db.user_follower.update_many({}, {'$set': {'read': '0'}})
    docs = db.user_follower.find({'read': '0'}, {'user_id': 1, 'follower_user_id': 1})
    counts = cnt.Counter()
    for doc in docs: 
        counts[doc['user_id']] += 1
        counts[doc['follower_user_id']] += 1
        db.user_follower.update_one({'_id': doc['_id']}, {'$set': {'read': '1', 'expired': '0'}})
    with open('./main/counts', 'wb') as file:
        pickle.dump(counts, file)
    return integer_encoding(mongo_client, db, counts)

def integer_encoding(mongo_client, db, counts):
    dictionary = {}
    for code, count in enumerate(counts.most_common()):
        dictionary[count[0]] = code
    with open('./main/dictionary', 'wb') as file:
        pickle.dump(dictionary, file)
    reversed_dictionary = {v: k for k, v in dictionary.items()}
    with open('./main/reversed_dictionary', 'wb') as file:
        pickle.dump(reversed_dictionary, file)

if __name__ == "__main__":
    reset_counter(mongo_client, db)