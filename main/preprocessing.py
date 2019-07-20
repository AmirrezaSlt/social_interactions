import collections 
from env import db
from helper_functions import timer, debug
import pickle

@debug
@timer
def counter(db):
    db.user_follower.update_many({'read': {'$exists': False}}, {'$set': {'read': '0'}})
    try: 
        with open('./main/counts', 'rb') as f:
            counts = pickle.load(f)
    except FileNotFoundError:
        return reset_counter(db)
    docs = db.user_follower.find({'read': '0'}, {'user_id': 1, 'follower_user_id': 1})
    for doc in docs: 
        counts[doc['user_id']] += 1
        counts[doc['follower_user_id']] += 1
        db.user_follower.update_one({'_id': doc['_id']}, {'$set': {'read': '1', 'expired': '0'}})
    with open('./main/counts', 'wb') as file:
        pickle.dump(counts, file)
    return integer_encoding(db, counts)

@debug
@timer
def reset_counter(db): 
    db.user_follower.update_many({}, {'$set': {'read': '0', 'expired': '0', 'visited': '0'}})
    docs = db.user_follower.find({'read': '0'}, {'user_id': 1, 'follower_user_id': 1})
    counts = collections.Counter()
    for doc in docs: 
        counts[doc['user_id']] += 1
        counts[doc['follower_user_id']] += 1
        db.user_follower.update_one({'_id': doc['_id']}, {'$set': {'read': '1', 'expired': '0'}})
    with open('./main/counts', 'wb') as file:
        pickle.dump(counts, file)
    return integer_encoding(db, counts)

@debug
@timer
def integer_encoding(db, counts):
    dictionary = {}
    for code, count in enumerate(counts.most_common()):
        dictionary[count[0]] = code
    with open('./main/dictionary', 'wb') as file:
        pickle.dump(dictionary, file)
    reversed_dictionary = {v: k for k, v in dictionary.items()}
    with open('./main/reversed_dictionary', 'wb') as file:
        pickle.dump(reversed_dictionary, file)
    return input_transform(db, dictionary)

@debug
@timer
def input_transform(db, dictionary):
    docs = db.user_follower.find({'read': '1', 'expired': '1', 'visited': '0'}, {'user_id': 1, 'follower_user_id': 1})
    for doc in docs: 
        db.user_follower.update_one({'_id': doc['_id']}, 
                {'$set': {'coded_user_id': dictionary[doc['user_id']], 'coded_follower_user_id': dictionary[doc['follower_user_id']]}})
    return graph_generator(db, dictionary)

@debug
@timer
def graph_generator(db, dictionary):
    docs = db.user_follower.find({'read': '1', 'expired': '0'}, {'user_id': 1, 'follower_user_id': 1})
    for doc in docs: 
        db.follower_graph.update_one({'user_id': dictionary[doc['user_id']]}, {'$push': {'followers': dictionary[doc['follower_user_id']]}}, upsert=True)
        db.user_follower.update_one({'_id': doc['_id']}, {'$set': {'expired': '1'}})        

if __name__ == "__main__":
    reset_counter(db)