#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


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