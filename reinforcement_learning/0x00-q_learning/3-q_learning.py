#!/usr/bin/env python3


"""contains epsilon_greedy function """

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ uses epsilon-greedy to determine the next action"""
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(Q.shape[1])
    return action


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ performs Q-learning"""
    max_epsilon = epsilon
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        rewards_current_episode = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if done is True and reward == 0:
                reward = -1
            Q[state, action] = Q[state, action] * \
                (1 - alpha) + alpha * (reward +
                                       gamma * np.max(Q[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done is True:
                break
        total_rewards.append(rewards_current_episode)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)
    return Q, total_rewards
