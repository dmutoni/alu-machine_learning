#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """[summary]

    Args:
        desc ([type], optional): [description]. Defaults to None.
        map_name ([type], optional): [description]. Defaults to None.
        is_slippery (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    return gym.make('FrozenLake-v0',
                    desc=desc, map_name=map_name,
                    is_slippery=is_slippery)
