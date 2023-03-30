# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/17 9:58
@Auth ： Yuhang Zhou
@File ：main.py
@IDE ：PyCharm
@Motto:nothing is required for work, just do it.
"""
import torch
import gym
from net import DQN
from trainer import train_dqn
from life.utils.replay.replay_buffer import ReplayBuffer
from wrapers import make_env
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from net import CNN2
import time
import joblib

env = gym.make("PongNoFrameskip-v4")
env = make_env(env)  # 注意这一步很关键！
state_dim = env.observation_space.shape
action_dim = env.action_space.shape or env.action_space.n
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

base_net = CNN2(
    in_channels=4,
    n_actions=6
)
replay_buffer = ReplayBuffer(capacity=10000)

agent_dqn = DQN(
    state_dim=state_dim,
    hidden_dim=128,
    action_dim=action_dim,
    learning_rate=0.0001,  # 降低学习率，下一步是改变模型，再下一步是使用tianshou
    gamma=0.92,
    epsilon=0.01,
    target_update=50,  # 从10增大为50
    device=device,
    q_net=base_net,
)
start_t = time.time()
result, agent = train_dqn(
    agent=agent_dqn,
    env=env,
    replay_buffer=replay_buffer,
    minimal_size=500,
    num_episodes=1400,  # 增大了迭代次数, 原来为500
    batch_size=64,  # batch_size从之前的128减少到了64，在gpu上的训练速度反而加快了
    return_agent=True
)

print(result)
print("training finished,using {} min.".format(time.time() - start_t))

joblib.dump(result, "./results/dqn1400iter_result_list.dat")
joblib.dump(agent, "./results/dqn1400iter_agent.dat")
