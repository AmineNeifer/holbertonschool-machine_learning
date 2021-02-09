#!/usr/bin/env python3


""" contains epsilon_greedy, train and play funcitons"""


def play(env, Q, max_steps=100):
    """has the trained agent play an episode"""
    env.reset()
    state = 0
    env.render()
    for i in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            break
    return reward
