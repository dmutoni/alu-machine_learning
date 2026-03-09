#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def policy(matrix, weight):
    """[summary]

    Args:
        matrix ([type]): [description]
        weight ([type]): [description]

    Returns:
        [type]: [description]
    """
    z = matrix.dot(
        weight)

    exp = np.exp(z)
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """[summary]

    Args:
        state ([type]): [description]
        weight ([type]): [description]
    """
    def softmax_grad(softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s
                           ) - np.dot(s, s.T)
    prb = policy(state, weight)

    action = np.argmax(prb)
    dsoftmax = softmax_grad(
        prb)[action, :]

    dlog = dsoftmax / prb[0, action]

    gradient = state.T.dot(dlog[None, :])
    return (action, gradient)
