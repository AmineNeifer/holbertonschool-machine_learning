#!/usr/bin/env python3


""" monte carlo method"""
import numpy as np


def play(env, p, max_steps=100):
    """has the trained agent play an episode"""
    episode = []
    state = env.reset()
    for _ in range(max_steps):
        action = p(state)
        new_state, reward, done, info = env.step(action)
        episode.append((state, reward))
        state = new_state
    return episode


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    performs the Monte Carlo algorithm:

    @env is the openAI environment instance
    @V is a numpy.ndarray of shape (s,) containing the value estimate
    @policy is a function that takes in a state and returns the next action
    @episodes is the total number of episodes to train over
    @max_steps is the maximum number of steps per episode
    @alpha is the learning rate
    @gamma is the discount rate

    Returns: V, the updated value estimate
    """
    n = env.observation_space.n
    discounts = np.array([gamma ** i for i in range(max_steps)])
    for _ in range(episodes):
        episode = play(env, policy, max_steps)
        np_ep = np.array(episode)
        for i in range(len(episode)):
            G = sum(np_ep[i:, 1] * discounts[:len(np_ep[i:, 1])])
            V[episode[i][0]] = V[episode[i][0]] + \
                alpha * (G - V[episode[i][0]])
    return V
