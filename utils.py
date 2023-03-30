# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/18 20:12
@Auth ： Yuhang Zhou
@File ：utils.py
@IDE ：PyCharm
@Motto:nothing is required for work, just do it.
"""
import numpy as np
import torch


def get_state(obs):
    """调整gym返回的state的维度"""
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    return state
