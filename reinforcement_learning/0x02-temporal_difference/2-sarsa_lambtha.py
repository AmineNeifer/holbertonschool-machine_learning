#!/usr/bin/env python3

import numpy as np


def epsilon_greedy(state, Q, epsilon):
    """ uses epsilon-greedy to determine the next action"""
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, int(Q.shape[1]))
    return(action)


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs SARSA(λ)

    @env: is the openAI environment instance
    @Q: is a numpy.ndarray of shape (state,a) containing the Q table
    @lambtha: is the eligibility trace factor
    @episodes: is the total number of episodes to train over
    @max_steps: is the maximum number of steps per episode
    @alpha: is the learning rate
    @gamma: is the discount rate
    @epsilon: is the initial threshold for epsilon greedy
    @min_epsilon: is the minimum value that epsilon should decay to
    @epsilon_decay: is the decay rate for updating epsilon between episodes

    Returns: Q
    """
    init_epsilon = epsilon
    Et = np.zeros((Q.shape))
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(state, Q, epsilon=epsilon)
        for _ in range(max_steps):
            Et = Et * lambtha * gamma
            Et[state, action] += 1.0
            new_state, reward, done, _ = env.step(action)
            new_action = epsilon_greedy(new_state, Q, epsilon=epsilon)
            delta_t = reward + gamma * \
                Q[new_state, new_action] - Q[state, action]
            Q[state, action] = Q[state, action] + \
                alpha * delta_t * Et[state, action]
            if done:
                break
            state = new_state
            action = new_action
        epsilon = min_epsilon + (init_epsilon - min_epsilon) *\
            np.exp(-epsilon_decay * episode)
    return Q
