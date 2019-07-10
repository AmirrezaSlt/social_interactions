import collections
from env import logger, mongo_client, db
from helper_functions import timer, debug, exception
import numpy as np
import pandas as pd
import pickle


@debug
@timer
def graph_generator():
    with mongo_client.start_session():
        query = db.user_follower.find({}, {'_id': 0, 'user_id': 1, 'follower_user_id': 1})
        db.user_interaction_graph.delete_many({})
    query_df = pd.DataFrame(list(query))
    ids = pd.concat([query_df.user_id, query_df.follower_user_id])
    output = query_df.groupby('user_id')['follower_user_id'].apply(list)
    documents = []
    for i in range(output.size):
        document = {'user_id': output.index[i], 'followers': output[i]}
        documents.append(document)
    db.user_interaction_graph.insert_many(documents)


@debug
@timer
def word_generator(sentence_count, sentence_length):
    with mongo_client.start_session():
        db.user_interaction_vocabulary.delete_many({})
        query = db.user_follower.find({}, {'_id': 0, 'user_id': 1, 'follower_user_id': 1})
        query_df = pd.DataFrame(list(query))
        # ids = pd.concat([query_df.user_id, query_df.follower_user_id])
        graph = query_df.groupby('user_id')['follower_user_id'].apply(set)
        for i in range(sentence_count):
            word = pd.Series()
            origin = np.random.choice(graph.index.values, 1)
            origin = pd.Series(origin)
            for j in range(sentence_length):
                word = word.append(origin)
                np.random.choice(graph[origin].values, 1)
                origin = np.random.choice(graph.index.values, 1)
                origin = pd.Series(origin)
            word.reset_index(drop=True, inplace=True)
            word.index = word.index.astype(str)
            word = word.to_dict()
            # document = {'path_number': i+1, 'path': word}
            db.user_interaction_vocabulary.insert_one({'words': word})


@debug
@timer
def build_dictionary():
    count = collections.Counter()
    with mongo_client.start_session():
        sentences = db.user_interaction_vocabulary.find({}, {'words': 1, '_id': 0})
        for sentence in sentences:
            sentence = sentence['words']
            for word in sentence:
                count[sentence[word]] += 1
        dictionary = {}
    for word in count:
        dictionary[word] = len(dictionary)
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # Zip is used for the later unk additions
    return count, reversed_dictionary, dictionary


@debug
@timer
def encode_data(dictionary, sentence_length):
    with mongo_client.start_session():
        sentences = db.user_interaction_vocabulary.find({})
        for sentence in sentences:
            id = sentence['_id']
            sentence = sentence['words']
            counter = 0
            codes = {}
            for i in range(sentence_length):
                code = dictionary[sentence[str(i)]]
                codes[str(i)] = code
                counter += 1
            db.user_interaction_vocabulary.update_one({'_id': id}, {'$set': {'codes': codes}}, upsert=False)


sentence_count = 3000
sentence_length = 1000      
# graph_generator()
# word_generator(sentence_count, sentence_length)
# count, reversed_dictionary, dictionary = build_dictionary()
# with open('dictionary', 'wb') as file:
#     pickle.dump(dictionary, file)
# with open('reversed_dictionary', 'wb') as file:
#     pickle.dump(reversed_dictionary, file)
# encode_data(dictionary, sentence_length)
