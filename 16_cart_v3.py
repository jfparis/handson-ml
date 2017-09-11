# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import gym
import tensorflow as tf
from datetime import datetime
import math

from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

env = gym.make("CartPole-v0")

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros_like(rewards)
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_reward(rewards, discount_rate):
    discounted_rewards = discount_rewards(rewards, discount_rate)
    reward_mean = discounted_rewards.mean()
    reward_std = discounted_rewards.std()
    return (discounted_rewards - reward_mean)/reward_std

def log_dir(prefix="", date=True):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run" + "" if not date else "-" + now
    return "{}/{}/".format(root_logdir, name)

# 1. Specify the neural network architecture
n_inputs = 4 # == env.observation_space.shape[0]
n_hidden = 4 # it's a simple task, we don't need more hidden neurons
n_outputs = 1 # only outputs the probability of accelerating left
initializer = tf.contrib.layers.variance_scaling_initializer()
learning_rate = 0.01

# 2. Build the neural network
state_in = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(state_in, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)

reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
f_action_holder = tf.cast(action_holder,tf.float32)

cross_entropy = f_action_holder*tf.log(outputs) + (1-f_action_holder)*tf.log(1-outputs)
loss = -tf.reduce_mean(cross_entropy*reward_holder)

optimizer = tf.train.AdamOptimizer(learning_rate)

grads_and_vars = optimizer.compute_gradients(loss)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []

for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

# Create the global step for monitoring the learning_rate and training.
global_step = get_or_create_global_step()

training_op = optimizer.apply_gradients(grads_and_vars_feed, global_step=global_step)

avg_reward = tf.placeholder(tf.float32)
max_reward = tf.placeholder(tf.float32)
min_reward = tf.placeholder(tf.float32)

avg_reward_summary = tf.summary.scalar("average_reward", avg_reward)
max_reward_summary = tf.summary.scalar("max_reward", max_reward)
min_reward_summary = tf.summary.scalar("min_reward", min_reward)
my_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_iterations = 2000
n_max_steps = 1000
n_games_per_update = 10
discount_rate = 0.95

render = False

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

sv = tf.train.Supervisor(logdir=log_dir("cart_v3", False), summary_op=None, init_fn=None)

with sv.managed_session(config=config) as sess:

    start_iteration = math.floor(tf.train.global_step(sess, global_step))

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    for iteration in range(start_iteration, n_iterations):
        print("Epoch", iteration)

        all_rewards = []  # all sequences of raw rewards for each episode - used for stats

        for game in range(n_games_per_update):
            current_rewards = []  # all raw rewards from the current episode
            current_gradients = []  # all gradients from the current episode
            ep_history = []
            obs = env.reset()
            if render:
                env.render()
            for step in range(n_max_steps):
                action_val = sess.run(outputs, feed_dict={state_in: obs.reshape(1, n_inputs)})  # one obs
                a = 1 if np.random.uniform() < action_val else 0
                y = 1 if a == 0 else 0
                obs1, reward, done, info = env.step(a)
                ep_history.append([obs, y, reward, obs1])
                obs = obs1
                current_rewards.append(reward)
                if render:
                    env.render()
                if done:
                    break

            all_rewards.append(current_rewards)
            # compute gradients for the most recent episode
            ep_history = np.array(ep_history)
            ep_history[:, 2] = discount_and_normalize_reward(ep_history[:, 2], discount_rate)
            feed_dict = {reward_holder: ep_history[:, 2],
                         action_holder: ep_history[:, 1], state_in: np.vstack(ep_history[:, 0])}
            grads = sess.run(gradients, feed_dict=feed_dict)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad/n_games_per_update

        rewards_outcome = [np.sum(reward) for reward in all_rewards]

        # At this point we have run the policy for 10 episodes, and we are
        # ready for a policy update using the algorithm described earlier.

        print("upgrading gradients")

        feed_dict = dictionary = dict(zip(gradient_placeholders, gradBuffer))
        _ = sess.run(training_op, feed_dict=feed_dict)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        summaries = sess.run(my_summary_op, feed_dict={avg_reward: np.mean(rewards_outcome), max_reward: np.max(rewards_outcome), min_reward: np.min(rewards_outcome)})
        sv.summary_computed(sess, summaries)


        sv.saver.save(sess, sv.save_path)