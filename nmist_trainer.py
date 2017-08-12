import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import os.path

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = 75
n_outputs = 10
learning_rate = 0.01
nb_epoch = 100
batch_size = 50

checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_deep_mnist_model"

three_layers = True

mnist = input_data.read_data_sets("datasets/")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")
X_valid = mnist.validation.images
y_valid = mnist.validation.labels.astype("int")

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    if three_layers:
        hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.relu)
        logits = tf.layers.dense(hidden3, n_outputs, name="logits")
    else:
        logits = tf.layers.dense(hidden2, n_outputs, name="logits")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("admin"):
    from datetime import datetime

    def log_dir(prefix="", root_logdir="tf_logs"):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "tf_logs"
        if prefix:
            prefix += "-"
        name = prefix + "run-" + now
        return "{}/{}/".format(root_logdir, name)
    filewriter = tf.summary.FileWriter(log_dir("mnist_dnn"), tf.get_default_graph())
    saver = tf.train.Saver()
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('log_accuracy', accuracy)

init = tf.global_variables_initializer()

with tf.Session() as session:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path,"rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(session, checkpoint_path)
    else:
        init.run()
        start_epoch = 0
        
    nb_batch = mnist.train.num_examples // batch_size
    for epoch in range(start_epoch, nb_epoch):
        for iteration in range(nb_batch):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            session.run(training_op, feed_dict={X: X_batch, y: y_batch}) # TBN
        loss_val, accuracy_val,  loss_summary_str, acc_summary_str = session.run([loss, accuracy, loss_summary, accuracy_summary], feed_dict={X: X_valid, y: y_valid})   
        filewriter.add_summary(loss_summary_str,epoch)
        filewriter.add_summary(acc_summary_str,epoch)
        if epoch % 5 == 0:
            print("epoch:", epoch, "loss", loss_val, "accuracy", accuracy_val)
            saver.save(session, checkpoint_path)
            with open(checkpoint_epoch_path,"wb") as f:
                f.write(b"%d" % (epoch + 1))
        
    # save final model
    saver.save(session, final_model_path)
    os.remove(checkpoint_epoch_path)