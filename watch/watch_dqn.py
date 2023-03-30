# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/22 18:37
@Auth ： Yuhang Zhou
@File ：watch_dqn.py
@IDE ：PyCharm
@Motto:nothing is required for work, just do it.
"""
import gym
import joblib
from wrapers import make_env
from utils import get_state
import time

env = gym.make("PongNoFrameskip-v4")
env = make_env(env)
agent = joblib.load("../results/dqn1400iter_agent.dat")

for i in range(2):
    done = False
    state = env.reset()
    state = get_state(state)
    while not done:
        act = agent.take_action(state)
        print("NOW ACTION : ", act)
        next_state, reward, done, _ = env.step(act)
        next_state = get_state(next_state)
        state = next_state
        env.render()
        time.sleep(0.01)
