#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def q_init(env):
    """[summary]

    Args:
        env ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.zeros((env.observation_space.n,
                     env.action_space.n))
