from agent import RLAgent
from environment import Environment
from config import config2
from plot import plot_reward
import nel

from collections import deque
import random
# import cPickle
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt


def center_painter_on_agent(painter, agent):
    position = agent.position()
    painter.set_viewbox(
        (position[0] - 70, position[1] - 70),
        (position[0] + 70, position[1] + 70))



def multiagent_test(num_agents):

    painters = list()
    agents = list()

    state_size = (config2.vision_range*2 + 1)**2 * config2.color_num_dims + config2.scent_num_dims
    
    #initialize agents
    for i in range(num_agents):

        env = Environment(config2)

        agent = RLAgent(env, state_size=state_size)

        #load one model per agent
        agent._load("outputs/models/NELQ_"+str(i)+"0")

        position = agent.position()
        painter = nel.MapVisualizer(env.simulator, config2,
            (position[0] - 70, position[1] - 70),
            (position[0] + 70, position[1] + 70))

        agents.append(agent)
        painters.append(painter)

    spawned_agents = 1

    for i in range(1000):

        for p in range(spawned_agents):
            
            s1 = agents[p].get_state()
            action, reward = agents[p].step(epsilon=0.1)

            center_painter_on_agent(painters[p], agents[p])
            painters[p].draw()

            if i == round(1000*spawned_agents)/len(agents) - 1:
                spawned_agents += 1
                    
def main():

    multiagent_test(2)

if __name__ == '__main__':
    main()
