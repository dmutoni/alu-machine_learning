#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def play(env, Q, max_steps=100):
    """[summary]

    Args:
        env ([type]): [description]
        Q ([type]): [description]
        max_steps (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    sta = env.reset()
    env.render()
    for _ in range(max_steps):
        a = np.argmax(Q[sta, :])
        sta, res, ok_, _ = env.step(a)
        env.render()
        if ok_:
            break
    return res
