# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from datetime import datetime
import os.path

import tensorflow as tf
import numpy as np

he_init = tf.contrib.layers.variance_scaling_initializer()


def leaky_relu(alpha=0.01):
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)

    return parametrized_leaky_relu


class DNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_hidden_layers=5, n_neurons=100, optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.01, batch_size=20, activation=tf.nn.elu, initializer=he_init, batch_normalisation = False,
                 dropout_rate = 0, random_state = None):
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self._session = None
        self.max_checks_without_progress = 20
        self.random_state = random_state
        self.batch_normalisation = batch_normalisation
        self.dropout_rate = dropout_rate

    def _log_dir(self, prefix=""):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "tf_logs"
        if prefix:
            prefix += "-"
        name = prefix + "run-" + now
        return "{}/{}/".format(root_logdir, name)

    def close_session(self):
        if self._session:
            self._session.close()

    def _build_dnn(self, inputs, scope_name):
        with tf.variable_scope(scope_name, "dnn"):

            training = tf.placeholder_with_default(False, shape=(), name='training')
            inputs = tf.layers.dropout(inputs, self.dropout_rate, training=training)

            for i in range(self.n_hidden_layers):
                inputs = tf.layers.dense(inputs, self.n_neurons, kernel_initializer = self.initializer, name="layer"+str(i))
                if self.batch_normalisation:
                    inputs = tf.layers.batch_normalization(inputs, training=training, momentum=0.9)
                inputs = self.activation(inputs)
                inputs = tf.layers.dropout(inputs, self.dropout_rate, training=training)

            self._training_flag =  training
            return inputs

    def _build_graph(self, n_inputs, n_outputs):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int64, shape=(None), name="y")

        dnn_outputs = self._build_dnn(X, "DNN")

        logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

        optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        training_op = optimiser.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self._X = X
        self._y = y
        self._Y_proba = Y_proba
        self._loss = loss
        self._training_op = training_op
        self._correct = correct
        self._accuracy = accuracy
        self._init = init
        self._saver = saver

    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving to disk)"""
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping, faster than loading from disk)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
        """Fit the model to the training set. If X_valid and y_valid are provided, use early stopping."""

        self.close_session()

        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)

        # Translate the labels vector to a vector of sorted class indices, containing
        # integers from 0 to n_outputs - 1.
        # For example, if y is equal to [8, 8, 9, 5, 7, 6, 6, 6], then the sorted class
        # labels (self.classes_) will be equal to [5, 6, 7, 8, 9], and the labels vector
        # will be translated to [3, 3, 4, 0, 2, 1, 1, 1]
        self.class_to_index_ = {label: index
                                for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label]
                      for label in y], dtype=np.int32)

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            log_path = self._log_dir()
            filewriter = tf.summary.FileWriter(log_path, tf.get_default_graph())
            loss_str = tf.summary.scalar('loss', self._loss)
            accuracy_str = tf.summary.scalar('accuracy', self._accuracy)

        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as session:
            self._init.run()
            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    session.run([self._training_op, extra_update_ops], feed_dict={self._X: X_batch, self._y: y_batch, self._training_flag: True})

                if X_valid is not None and y_valid is not None:
                    acc_sum_val, loss_sum_val, accuracy_val, loss_val = session.run([accuracy_str, loss_str, self._accuracy, self._loss ], feed_dict = {self._X: X_valid, self._y: y_valid})
                    filewriter.add_summary(loss_sum_val, epoch)
                    filewriter.add_summary(acc_sum_val,epoch)
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress +=1
                        if checks_without_progress > self.max_checks_without_progress:
                            print("Early stopping!")
                            break
                    print("Epoch:", epoch, "loss:", loss_val, "accuracy:",accuracy_val)

            if best_params:
                self._restore_model_params(best_params)
                acc_test = self._accuracy.eval(feed_dict={self._X: X_valid, self._y: y_valid})
                print("Final test accuracy: {:.2f}%".format(acc_test * 100))


        return self

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)


def test():
    # load the training set
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("datasets/")
    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    train_indices = [i for i in range(len(y_train)) if y_train[i] <= 4]
    test_indices = [i for i in range(len(y_test)) if y_test[i] <= 4]

    X_train4 = X_train[train_indices]
    X_test4 = X_test[test_indices]
    y_train4 = y_train[train_indices]
    y_test4 = y_test[test_indices]


    dnc = DNNClassifier(batch_size=500, n_neurons=140,  activation=leaky_relu(alpha=0.1), learning_rate=0.01, random_state=42,
                        batch_normalisation = False, dropout_rate=0.5)
    dnc.fit(X_train4, y_train4, 100, X_test4, y_test4)


def main():
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("datasets/")
    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    train_indices = [i for i in range(len(y_train)) if y_train[i] <= 4]
    test_indices = [i for i in range(len(y_test)) if y_test[i] <= 4]

    X_train4 = X_train[train_indices]
    X_test4 = X_test[test_indices]
    y_train4 = y_train[train_indices]
    y_test4 = y_test[test_indices]

    param_distribs = {
        "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
        "batch_size": [10, 50, 100, 500],
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
        # you could also try exploring different numbers of hidden layers, different optimizers, etc.
        # "n_hidden_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        # "optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95)],
    }

    rnd_search = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,
                                    fit_params={"X_valid": X_test4, "y_valid": y_test4, "n_epochs": 1000}, verbose=2)
    rnd_search.fit(X_train4, y_train4)

#if __name__ == "__main__":
test()