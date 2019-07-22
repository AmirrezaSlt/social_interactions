import collections 
from env import db
from helper_functions import timer, debug
import pickle

# TODO: check for consistency in update calls 
# TODO: Perform case_by_case parrallel operations based on DAGs for this data (better with golang or ...)
# In case of new API call: read:0, encoded:0, visited:0, graph:1
@timer
def counter(db):
    try: 
        with open('./main/counts', 'rb') as f:
            counts = pickle.load(f)
    except FileNotFoundError:
        return reset_counter(db)
    docs = db.user_follower.find({'read': '0'}, {'user_id': 1, 'follower_user_id': 1})
    for doc in docs: 
        counts[doc['user_id']] += 1
        counts[doc['follower_user_id']] += 1
        db.user_follower.update_one({'_id': doc['_id']}, {'$set': {'read': '1'}})
    with open('./main/counts', 'wb') as file:
        pickle.dump(counts, file)
    return integer_encoding(db, counts)

@timer
def reset_counter(db): 
    db.user_follower.update_many({}, {'$set': {'read': '0', 'encoded': '0', 'graph': '0','visited': '0'}})
    db.graph.drop()
    docs = db.user_follower.find({'read': '0'}, {'user_id': 1, 'follower_user_id': 1})
    counts = collections.Counter()
    for doc in docs: 
        counts[doc['user_id']] += 1
        counts[doc['follower_user_id']] += 1
        db.user_follower.update_one({'_id': doc['_id']}, {'$set': {'read': '1'}})
    with open('./main/counts', 'wb') as file:
        pickle.dump(counts, file)
    return integer_encoding(db, counts)

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

@timer
def input_transform(db, dictionary):
    docs = db.user_follower.find({'read': '1', 'encoded': '0'}, {'user_id': 1, 'follower_user_id': 1})
    for doc in docs: 
        db.user_follower.update_one({'_id': doc['_id']}, 
                {'$set': {'encoded': '1','coded_user_id': dictionary[doc['user_id']],
                 'coded_follower_user_id': dictionary[doc['follower_user_id']]}})
    return graph_generator(db, dictionary)

@timer
def graph_generator(db, dictionary):
    docs = db.user_follower.find({'encoded': '1', 'graph': '0'}, {'coded_user_id': 1, 'coded_follower_user_id': 1}, batch_size=10)
    for doc in docs: 
        db.follower_graph.update_one({'user_id': doc['coded_user_id']}, 
            {'$push': {'followers': doc['coded_follower_user_id']}}, upsert=True)
        db.user_follower.update_one({'_id': doc['_id']}, {'$set': {'graph': '1'}})        

if __name__ == "__main__":
    # reset_counter(db)
    with open('./main/dictionary', 'rb') as f:
        dictionary = pickle.load(f)
    graph_generator(db, dictionary)

# for batching: for loop strings --> for loop in each string --> for loop for the repeats --> max(0, index - window)< < min(max_path, index + window)