# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
from datetime import datetime
import os.path

import tensorflow as tf
import numpy as np

n_inputs = 28 * 28  # MNIST
n_outputs = 5
layers_shape = [100]*5
layers_activation = [tf.nn.elu] *5
he_init = tf.contrib.layers.variance_scaling_initializer()
layers_init = [he_init] *5
learning_rate = 0.01
nb_epoch = 500
batch_size = 20
max_checks_without_progress = 20



def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

def build_dnn(inputs, shape, activation, init, scope_name):
    with tf.variable_scope(scope_name, "dnn"):
        for i in range(len(layers_shape)):
            inputs = tf.layers.dense(inputs, shape[i], activation=activation[i], kernel_initializer = init, name="layer"+str(i))

        return inputs

def build_graph():
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    dnn_outputs = build_dnn(X, layers_shape, layers_activation, he_init, "DNN")

    logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimiser.minimize(loss)

    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    return X, y, Y_proba, loss, training_op, correct, accuracy

def train():
    X, y, Y_proba, loss, training_op, correct, accuracy = build_graph()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    log_path = log_dir()
    filewriter = tf.summary.FileWriter(log_path, tf.get_default_graph())
    loss_str = tf.summary.scalar('loss', loss)
    accuracy_str = tf.summary.scalar('accuracy', accuracy)
    checks_without_progress = 0 
    best_loss = np.infty

    # load the training set
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("datasets/")
    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    train_indices = [i for i in range(len(y_train)) if y_train[i]<=4]
    test_indices = [i for i in range(len(y_test)) if y_test[i]<=4]
    
    X_train4 = X_train[train_indices]
    X_test4 = X_test[test_indices]
    y_train4 = y_train[train_indices]
    y_test4 = y_test[test_indices]

    with tf.Session() as session:
        init.run()
        for epoch in range(nb_epoch):
            rnd_idx = np.random.permutation(len(X_train4))
            for rnd_indices in np.array_split(rnd_idx, len(X_train4) // batch_size): 
                X_batch, y_batch = X_train4[rnd_indices], y_train4[rnd_indices]
                session.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
            acc_sum_val, loss_sum_val, accuracy_val, loss_val = session.run([accuracy_str, loss_str, accuracy, loss ], feed_dict = {X: X_test4, y: y_test4})
            filewriter.add_summary(loss_sum_val, epoch)
            filewriter.add_summary(acc_sum_val,epoch)
            if loss_val < best_loss:
                save_path = saver.save(session, "./my_mnist_model_0_to_4.ckpt")
                best_loss = loss_val
                checks_without_progress = 0
            else:
                checks_without_progress +=1
                if checks_without_progress > max_checks_without_progress:
                    print("Early stopping!")
                    break
            print("Epoch:", epoch, "loss:", loss_val, "accuracy:",accuracy_val)

    with tf.Session() as session:
        saver.restore(session, "./my_mnist_model_0_to_4.ckpt")
        acc_test = accuracy.eval(feed_dict={X: X_test4, y: y_test4})
        print("Final test accuracy: {:.2f}%".format(acc_test * 100))

def main():
    train() 


#if __name__ == "__main__":
main()