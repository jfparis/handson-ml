# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
from datetime import datetime
import os.path

import tensorflow as tf
import numpy as np

he_init = tf.contrib.layers.variance_scaling_initializer()
learning_rate=0.01

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("datasets/")
X_train1 = mnist.train.images
y_train1 = mnist.train.labels

X_train2 = mnist.validation.images
y_train2 = mnist.validation.labels

X_test = mnist.test.images
y_test = mnist.test.labels


def leaky_relu(alpha=0.01):
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)

    return parametrized_leaky_relu

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

def build_dnn(inputs, scope_name, training, n_hidden_layers=5, n_neurons=100, dropout_rate=0, initializer = he_init, activation =leaky_relu()):
    with tf.variable_scope(scope_name, "dnn"):

        inputs = tf.layers.dropout(inputs, dropout_rate, training=training)

        for i in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs, n_neurons, kernel_initializer = initializer, name="layer"+str(i))
            inputs = activation(inputs)
            inputs = tf.layers.dropout(inputs, dropout_rate, training=training)

        return inputs

def build_graph(n_inputs, n_outputs):
    X = tf.placeholder(tf.float32, shape=(None, 2, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    X1, X2 = tf.unstack(X, axis=1)

    training = tf.placeholder_with_default(False, shape=(), name='training')

    dnn_outputs1 = build_dnn(X1, "DNN_A", training)
    dnn_outputs2 = build_dnn(X2, "DNN_B", training)

    dnn_outputs = tf.concat([dnn_outputs1, dnn_outputs2], axis =1)
    hidden_layer = tf.layers.dense(dnn_outputs, units=10, activation=leaky_relu(), kernel_initializer = he_init, name="hidden_summary")
    logits = tf.layers.dense(hidden_layer, units=1, kernel_initializer=he_init, name="logits")

    y_proba = tf.nn.sigmoid(logits)
    y_pred = tf.cast(tf.greater_equal(logits, 0), tf.int32)

    y_as_float = tf.cast(y, tf.float32)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_as_float, logits=logits)
    loss = tf.reduce_mean(xentropy)

    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimiser.minimize(loss)

    y_pred_correct = tf.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(y_pred_correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    return X, y, loss, training_op, accuracy ,init, saver

def generate_batch(images, labels, batch_size):
    size1 = batch_size // 2
    size2 = batch_size - size1
    if size1 != size2 and np.random.rand() > 0.5:
        size1, size2 = size2, size1
    X = []
    y = []
    while len(X) < size1:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(images), 2)
        if rnd_idx1 != rnd_idx2 and labels[rnd_idx1] == labels[rnd_idx2]:
            X.append(np.array([images[rnd_idx1], images[rnd_idx2]]))
            y.append([1])
    while len(X) < batch_size:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(images), 2)
        if labels[rnd_idx1] != labels[rnd_idx2]:
            X.append(np.array([images[rnd_idx1], images[rnd_idx2]]))
            y.append([0])
    rnd_indices = np.random.permutation(batch_size)
    return np.array(X)[rnd_indices], np.array(y)[rnd_indices]


def train():
    X_test1, y_test1 = generate_batch(X_test, y_test, batch_size=len(X_test))

    X, y, loss, training_op, accuracy, init, saver = build_graph(28*28, 1)

    n_epochs = 300
    batch_size = 500

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = generate_batch(X_train1, y_train1, batch_size)
                loss_val, _ = sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch})
            print(epoch, "Train loss:", loss_val)
            if epoch % 5 == 0:
                acc_test = accuracy.eval(feed_dict={X: X_test1, y: y_test1})
                print(epoch, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./my_digit_comparison_model.ckpt")

train()

