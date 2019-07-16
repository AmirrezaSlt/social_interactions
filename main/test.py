from __future__ import absolute_import, division, print_function

import math
import numpy as np
import tensorflow as tf


dictionary_size = 8
batch_size = 4
embedding_size = 2
num_sampled = 2


def generate_batch():
    global data_index
    final_batch = np.array([])
    final_labels = np.array([])
    for _ in range(batch_size):
        if data_index % 2 == 0:
            batch, labels = np.random.random_integers(low=0, high=3, size=2)
            while batch == labels:
                batch, labels = np.random.random_integers(low=0, high=3, size=2)
        if data_index % 2 == 1:
            batch, labels = np.random.random_integers(low=4, high=7, size=2)
            while batch == labels:
                batch, labels = np.random.random_integers(low=4, high=7, size=2)
        final_batch = np.append(final_batch, batch)
        final_labels = np.append(final_labels, labels)
    final_batch = final_batch.astype(int)
    final_labels = final_labels.astype(int)
    final_labels = np.reshape(final_labels,[batch_size, 1])
    data_index += 1
    return final_batch, final_labels
data_index = 0
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    with tf.device('/cpu:0'):
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([dictionary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([dictionary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([embedding_size]))
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=dictionary_size))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    init = tf.global_variables_initializer()
num_steps = 1001
with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch()
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        if step % 10 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step ', step, ': ', average_loss)
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

