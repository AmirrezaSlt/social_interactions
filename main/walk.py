from env import db
from helper_functions import timer, debug
import pickle
import numpy as np 

@timer
def walk(db, path_length):  
    docs = db.user_follower.find({'graph': '1', 'visited': '0'}, {'coded_user_id': 1, 'coded_follower_user_id': 1})
    for doc in docs: 
        path = walk_path(db, doc, path_length)
        db.paths.insert_one({'path': path.tolist()})
        # db.user_follower.update_one({'_id': doc['_id']}, {'$set': {'visited': '1'}})

@timer 
def walk_path(db, doc, path_length):
    wo = 3 # The weight of returning to the origin of the string
    wh = 2 # The weight of returning to nodes it has visited before (except the origin)
    wn = 1 # The weight of visiting new nodes  
    path = np.ones(path_length, dtype=int) * (-1)
    path[0] = doc['coded_user_id']
    path[1] = doc['coded_follower_user_id']
    for i in range(2, path_length): 
        print(str(path[i-1]))
        options = db.follower_graph.find_one({'user_id': path[i-1].item()}, {'_id': 0, 'followers': 1})
        if options is None:
            return path
        else:
            options = options['followers']
            weights = np.empty_like(options)
            for index, option in enumerate(options): 
                if option == path[0]: 
                    weights[index] = wo
                elif option in path:
                    weights[index] = wh
                else: 
                    weights[index] = wn
            path[i] = np.random.choice(options, 1, p=weights/np.sum(weights))      
    return path


if __name__ == "__main__": 
    walk(db, 10)