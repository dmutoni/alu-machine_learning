#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1,
          epsilon_decay=0.05):
    """
    performs Q-learning
    :param env: FrozenLakeEnv instance
    :param Q: numpy.ndarray containing the Q-table
    :param episodes: total number of episodes to train over
    :param max_steps: maximum number of steps per episode
    :param alpha: the learning rate
    :param gamma: the discount rate
    :param epsilon: the initial threshold for epsilon greedy
    :param min_epsilon: the minimum value that epsilon should
        decay to
    :param epsilon_decay: the decay rate for updating epsilon
        between episodes
    :return: Q, tf_ra
        Q is the updated Q-table
        tf_ra is a list containing the rewards per episode
    """
    
    e_copy = epsilon
    tf_ra = []
    for e in range(episodes):
        state = env.reset()
        reward_sum = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if done and reward == 0:
                reward = -1
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            state = new_state
            reward_sum += reward

            if done:
                break
        tf_ra.append(reward_sum)
        epsilon = (min_epsilon + (e_copy - min_epsilon) *
                   np.exp(-epsilon_decay * e))
    return Q, tf_ra

def epsilon_greedy(Q, state, epsilon):
    """[summary]

    Args:
        Q ([type]): [description]
        state ([type]): [description]
        epsilon ([type]): [description]

    Returns:
        [type]: [description]
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])
