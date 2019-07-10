from __future__ import absolute_import, division, print_function

import argparse
import collections
from data_preparation import sentence_length, sentence_count
from env import logger, mongo_client, db
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random

from numpy import recarray
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
import sys
from tempfile import gettempdir
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


with open('reversed_dictionary', 'rb') as f:
    reversed_dictionary = pickle.load(f)
with open('dictionary', 'rb') as f:
    dictionary = pickle.load(f)
dictionary_size = len(reversed_dictionary)
log_dir = 'Project/main/log'
data_size = sentence_length * sentence_count
batch_size = 4
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 2   # Number of negative examples to sample.
valid_size = 2  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size-1, replace=False)
valid_examples = np.append(valid_examples, dictionary['4f67936a91d408bf2a000002'])

# def one_hot_encode(data):
#     code = np.zeros([1, dictionary_size])
#     code[0, data] = 1
#     return code
#
#
# def generate_batch(batch_size, num_skips, skip_window):
#     global data_index
#     assert batch_size % num_skips == 0
#     assert num_skips <= 2 * skip_window
#     batch = np.ndarray(shape=(batch_size, dictionary_size), dtype=np.int32)
#     labels = np.ndarray(shape=(batch_size, dictionary_size), dtype=np.int32)
#     span = 2 * skip_window + 1  # [ skip_window target skip_window ]
#     buffer = collections.deque(maxlen=span)
#     # while (data_index + 1) // sentence_length != (data_index + 1 + span) // sentence_length:
#     while data_index // sentence_length > (sentence_length - span - 1) // sentence_length:
#         data_index += 1
#     if data_index + span > data_size:
#         data_index = 0
#     sentence_num = data_index // sentence_length
#     with mongo_client.start_session():
#         data = db.user_interaction_vocabulary.find({}, {'codes': 1, '_id': 0})[sentence_num]
#         data = data['codes']
#         for i in range(data_index, data_index + span):
#             buffer.append(data[str(i)])
#     data_index += span
#     for i in range(batch_size // num_skips):
#         context_words = [w for w in range(span) if w != skip_window]
#         words_to_use = random.sample(context_words, num_skips)
#         for j, context_word in enumerate(words_to_use):
#             batch[i * num_skips + j, :] = one_hot_encode(buffer[skip_window])
#             labels[i * num_skips + j, :] = one_hot_encode(buffer[context_word])
#     return batch, labels

#
# data_index = 0
# graph = tf.Graph()
# with graph.as_default():
#     with tf.name_scope('encoding'):
#         x = tf.placeholder(tf.float32, shape=(batch_size, dictionary_size))
#         y_label = tf.placeholder(tf.float32, shape=(batch_size, dictionary_size))
#         W1 = tf.Variable(tf.truncated_normal([dictionary_size, embedding_size]))
#         b1 = tf.Variable(tf.truncated_normal([embedding_size]))
#         hidden_representation = tf.add(tf.matmul(x, W1), b1)
#         hidden_representation = tf.Print(hidden_representation, [hidden_representation], 'encoding: ')
#
#     with tf.name_scope("decoding"):
#         W2 = tf.Variable(tf.truncated_normal([embedding_size, dictionary_size]))
#         b2 = tf.Variable(tf.truncated_normal([dictionary_size]))
#         reconstruction = tf.add(tf.matmul(hidden_representation, W2), b2)
#         prediction = tf.nn.softmax(reconstruction)
#         prediction = tf.Print(prediction, [prediction], 'decoding: ')
#     with tf.name_scope('loss'):
#         # loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
#         loss = tf.losses.mean_squared_error(
#             labels=y_label,
#             predictions=prediction
#         )
#         prediction = tf.Print(loss, [loss], 'loss: ')
#         # tf_loss_summary = tf.summary.scalar('loss', loss)
#         train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#
# with tf.Session(graph=graph) as sess:
#     # writer = tf.summary.FileWriter(log_dir, sess.graph)
#     init = tf.global_variables_initializer()
#     init.run()
#     epochs = 15
#     for _ in range(epochs):
#         x_train, y_train = generate_batch(batch_size, num_skips, skip_window)
#         # x_train = np.matrix(x_train[0, :])
#         # y_train = np.matrix(y_train[0, :])
#         _, loss_val = sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
#         print(loss_val)
#         # print(sess.run(W1, feed_dict={x: x_train, y_label: y_train}))
#         # print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
# WIP


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    # while (data_index + 1) // sentence_length != (data_index + 1 + span) // sentence_length:
    while data_index // sentence_length < (data_index + span) // sentence_length:
        data_index += 1
    if data_index + span > data_size:
        data_index = 0
    sentence_num = data_index // sentence_length
    with mongo_client.start_session():
        data = db.user_interaction_vocabulary.find({}, {'codes': 1, '_id': 0})[sentence_num]
        data = data['codes']
        for i in range(data_index % sentence_length, (data_index + span) % sentence_length):
            buffer.append(data[str(i)])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
    print('batch: ', batch)
    print('labels', labels)
    return batch, labels


data_index = 0
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
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
            nce_biases = tf.Variable(tf.zeros([dictionary_size]))
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=dictionary_size))
    tf.summary.scalar('loss', loss)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
num_steps = 100001
with tf.Session(graph=graph) as session:
    writer = tf.summary.FileWriter(log_dir, session.graph)
    init.run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        run_metadata = tf.RunMetadata()
        _, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict, run_metadata=run_metadata)
        average_loss += loss_val
        writer.add_summary(summary, step)
        if step == (num_steps - 1):
          writer.add_run_metadata(run_metadata, 'step%d' % step)
        if step % 10 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    with open(log_dir + '/metadata.tsv', 'w') as f:
        for i in range(dictionary_size):
            f.write(reversed_dictionary[i] + '\n')
    saver.save(session, os.path.join(log_dir, 'model.ckpt'))
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)
writer.close()


def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.savefig(filename)
    try:
        tsne = TSNE(
            perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)


def main(unused_argv):
  current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(current_path, 'log'),
      help='The log directory for TensorBoard summaries.')
  flags, unused_flags = parser.parse_known_args()
  word2vec_basic(flags.log_dir)


if __name__ == '__main__':
  tf.app.run()
