from env import db
import math
import numpy as np
import pickle
import tensorflow as tf


# ! Don't run when the paths are being generated!
with open('./main/dictionary', 'rb') as f:
    dictionary = pickle.load(f)
vocabulary_size = len(dictionary)
print(vocabulary_size)
batch_size = 4
embedding_size = 5
num_sampled = 2
skip_window = 2
class batch_generator:
    def __init__(self, db, db_batch_size, skip_window, batch_size):
        self.db_batch_size = db_batch_size
        self.db = db
        self.skip_window = skip_window
        self.path_index = 0
        self.word_index = 0
        self.batch_size = batch_size
        self.paths = db.paths.find({}, batch_size=self.db_batch_size)
        self.paths_count = db.paths.count_documents

    def __get_path(self):
        path = self.paths[self.path_index]
        path = path['path']
        path_length = len(path)
        return path, path_length

    def generate_batch(self):
        batch = np.zeros([self.batch_size], dtype=int)
        labels = np.zeros([self.batch_size, 1], dtype=int)
        path, path_length = self.__get_path()
        for batch_index in range(self.batch_size):
            batch[batch_index] = path[self.word_index]
            label_choices = list(range(max(0 , self.word_index - self.skip_window), min(path_length, self.word_index + self.skip_window + 1)))
            label_choices.remove(self.word_index)
            labels[batch_index] = path[np.random.choice(label_choices, 1).item()]
            self.word_index += 1
            if self.word_index == path_length:
                self.word_index = 0
                self.path_index += 1
                if self.path_index == self.paths_count: 
                    self.path_index = 0
                    self.paths = self.db.paths.find({}, batch_size=self.db_batch_size)
                path, path_length = self.__get_path()
        return batch, labels 

generator = batch_generator(db=db, db_batch_size=5, skip_window=skip_window, batch_size=batch_size)
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    with tf.device('/cpu:0'):
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    init = tf.global_variables_initializer()
num_steps = 100001
with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generator.generate_batch()
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        print(batch_inputs)
        print(batch_labels)
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        if step % 10 == 0:
            if step > 0:
                average_loss /= 10
            print('Average loss at step ', step, ': ', loss_val)
            # test_in = tf.constant([1., 0., 0., 0., 0., 0., 0., 0.], shape=[1, 8])
            # # print(nce_weights.eval())
            # embed = tf.add(tf.matmul(test_in, nce_weights), nce_biases)
            # print('1, ', embed.eval())
            # average_loss = 0
            # test_in = tf.constant([0., 1., 0., 0., 0., 0., 0., 0.], shape=[1, 8])
            # # print(nce_weights.eval())
            # embed = tf.add(tf.matmul(test_in, nce_weights), nce_biases)
            # print('2: ', embed.eval())
            # average_loss = 0
            # test_in = tf.constant([0., 0., 0., 0., 1., 0., 0., 0.], shape=[1, 8])
            # # print(nce_weights.eval())
            # embed = tf.add(tf.matmul(test_in, nce_weights), nce_biases)
            # print('5, ', embed.eval())
            # average_loss = 0
    final_embeddings = normalized_embeddings.eval()
