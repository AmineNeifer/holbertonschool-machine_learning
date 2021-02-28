#!/usr/bin/env python3


""" lambda method, similar to monte carlo"""
import gym
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm

    @env: openAI environment instance
    @V: numpy.ndarray of shape (s,) containing the value estimate
    @policy: returns the next action to take
    @episodes: is the total number of episodes to train over
    @max_steps: is the maximum number of steps per episode
    @alpha: is the learning rate
    @gamma: is the discount rate

    Returns: V
    """
    state = env.observation_space.n
    Et = np.zeros(state)
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            Et = Et * lambtha * gamma
            Et[state] += 1.0

            new_state, reward, done, info = env.step(policy(state))

            delta_t = reward + gamma * V[new_state] - V[state]
            V[state] = V[state] + alpha * delta_t * Et[state]

            if done:
                break
            state = new_state
    return V
