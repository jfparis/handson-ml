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
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
        for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std
        for discounted_rewards in all_discounted_rewards]

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
X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)

# 3. Select a random action based on the estimated probabilities
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

optimizer = tf.train.AdamOptimizer(learning_rate)

grads_and_vars = optimizer.compute_gradients(cross_entropy)
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
save_iterations = 10
discount_rate = 0.95

render = False

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

sv = tf.train.Supervisor(logdir=log_dir("cart", False), summary_op=None, init_fn=None)

with sv.managed_session(config=config) as sess:

    start_iteration = math.floor(tf.train.global_step(sess, global_step))

    for iteration in range(start_iteration, n_iterations):
        print("Epoch", iteration)
        print("gstep", tf.train.global_step(sess, global_step))
        all_rewards = []  # all sequences of raw rewards for each episode
        all_gradients = []  # gradients saved at each step of each episode

        for game in range(n_games_per_update):
            current_rewards = []  # all raw rewards from the current episode
            current_gradients = []  # all gradients from the current episode
            obs = env.reset()
            if render:
                env.render()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run([action, gradients],
                                                     feed_dict={X: obs.reshape(1, n_inputs)})  # one obs

                obs, reward, done, info = env.step(action_val[0][0])
                if render:
                    env.render()
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        # At this point we have run the policy for 10 episodes, and we are
        # ready for a policy update using the algorithm described earlier.

        rewards_outcome = [np.sum(reward) for reward in all_rewards]
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
            # multiply the gradients by the action scores, and compute the mean
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)],
                                     axis=0)
            feed_dict[grad_placeholder] = mean_gradients
        print("upgrading gradients")
        sess.run(training_op, feed_dict=feed_dict)

        summaries = sess.run(my_summary_op, feed_dict={avg_reward: np.mean(rewards_outcome), max_reward: np.max(rewards_outcome), min_reward: np.min(rewards_outcome)})
        sv.summary_computed(sess, summaries)


        sv.saver.save(sess, sv.save_path)