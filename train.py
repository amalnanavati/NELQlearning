from agent import *
from environment import Environment
from config import config2, agent_config, train_config
from plot import plot_reward
import nel

from collections import deque
import random
from six.moves import cPickle
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import os

import time

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def compute_td_loss(batch_size, agent, replay_buffer, gamma, optimizer):
    # Sample a random minibatch from the replay history.
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = agent.policy(state)
    q_values_target = agent.target(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = q_values_target.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    # loss = F.smooth_l1_loss(q_value,  Variable(expected_q_value.data))
    loss = F.mse_loss(q_value,  Variable(expected_q_value.data))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def plot(plot_agent):

    fig = plt.figure()
    colors=["blue","red","green","yellow"]

    chart_1 = fig.add_subplot(211)
    chart_1.set_title('sum of rewards')
    chart_1.grid(True)

    chart_2 = fig.add_subplot(212)
    chart_2.set_title('rewards per timestep')
    chart_2.grid(True)

    fig.tight_layout()

    for i in range(len(plot_agent)):


        #plot_data gets information about agents != 0
        plot_data = plot_agent[i] 

        #base_data gets data of agent 0
        base_data = plot_agent[0]

        #shift step is the step this agent spawned
        shift_step = plot_data[0][0] - base_data[0][0]

        #shift clock is the clock this agent spawned
        shift_clock = plot_data[1][0] - base_data[0][0]

        #do the shift
        for k in range(len(plot_data[0])):
            plot_data[0][k] -= shift_step
            plot_data[1][k] -= shift_clock 

        chart_1.plot(plot_data[0],plot_data[2],color=colors[i])
        chart_2.plot(plot_data[1],plot_data[2],color=colors[i])

    plt.show()

def get_epsilon(i, EPS_START, EPS_END, EPS_DECAY_START, EPS_DECAY_END):
    if i < EPS_DECAY_START:
        epsilon = EPS_START
    elif i > EPS_DECAY_END:
        epsilon = EPS_END
    else:
        epsilon = EPS_START - (EPS_START - EPS_END) * (i -
                                                       EPS_DECAY_START) / (EPS_DECAY_END - EPS_DECAY_START)
    return epsilon


def save_training_run(losses, rewards, agent, model_path):
    with open('outputs/train_stats.pkl', 'wb') as f:
        cPickle.dump((losses, rewards), f)

    agent.save(filepath=model_path)

def train(agent, env, actions):
    EPS_START = 1.
    EPS_END = .1
    EPS_DECAY_START = 1000.
    EPS_DECAY_END = 50000.

    def eps_func(i):
        return get_epsilon(i, EPS_START, EPS_END, EPS_DECAY_START, EPS_DECAY_END)
    num_steps_save_training_run = train_config['num_steps_save_training_run']
    policy_update_frequency = train_config['policy_update_frequency']
    target_update_frequency = train_config['target_update_frequency']
    eval_frequency = train_config['eval_frequency']
    batch_size = train_config['batch_size']
    training_steps = 0
    discount_factor = train_config['discount_factor']
    eval_steps = train_config['eval_steps']
    max_steps = train_config['max_steps']

    replay = list()
    tr_reward = list()
    all_rewards = list()
    rewards = list()
    loss = list()
    losses = list()
    eval_reward = list()
    model_path = list()
    p_path = list()
    optimizer = list()
    painter = list()
    plt_fn = list()
    save_fn = list()
    writer = list()

    now = time.time()

    for i in range(len(agent)):

        replay.append(ReplayBuffer(train_config['replay_buffer_capacity']))
        tr_reward.append(0)
        all_rewards.append(deque(maxlen=100))
        rewards.append([])
        losses.append([])
        loss.append(None)

        agent[i].update_target()

        optimizer.append(optim.Adam(agent[i].policy.parameters(),
            lr=agent_config['learning_rate']))

        f = open("outputs/plot_"+str(i)+".txt","w")
        writer.append(f)

    spawned_agents = 1

    for training_steps in range(max_steps):
        # Update current exploration parameter epsilon, which is discounted
        # with time.

        epsilon = eps_func(training_steps)

        if training_steps == int(round(max_steps*spawned_agents))/len(agent):
            print("ADDING AN AGENT", training_steps)
            spawned_agents+=1

        for i in range(spawned_agents):

            add_to_replay = len(agent[i].prev_states) >= 1

            # Get current state.
            s1 = agent[i].get_state()

            # Make a step.
            action, reward = agent[i].step(epsilon)

            # Update state according to step.
            s2 = agent[i].get_state()

            # Accumulate all rewards.
            tr_reward[i] += reward
            all_rewards[i].append(reward)
            rewards[i].append(np.sum(all_rewards[i]))

            # Add to memory current state, action it took, reward and new state.
            if add_to_replay:
                # enum issue in server machine
                replay[i].push(s1, action.value, reward, s2, False)

            # Update the network parameter every update_frequency steps.
            if training_steps % policy_update_frequency == 0:
                if batch_size < len(replay[i]):
                    # Compute loss and update parameters.
                    loss[i] = compute_td_loss(
                        batch_size, agent[i], replay[i], discount_factor, optimizer[i])
                    losses[i].append(loss[i].data[0])

            if training_steps % 200 == 0 and training_steps > int(round(max_steps*i))/len(agent):
                print('step = ', training_steps)
                print("loss_"+str(i)+" = ", loss[i].data[0])
                print("train reward_"+str(i)+" = ", tr_reward[i])
                print('')

                writer[i].write(str(training_steps)+" "+str(round(time.time()-now))+
                    " "+str(tr_reward[i])+" "+str(loss[i].data[0])+"\n")

            if training_steps % target_update_frequency == 0:
                agent[i].update_target()

            model_path.append('outputs/models/NELQ_'+ str(i) + str(training_steps))
            p_path.append('outputs/plots/NELQ_plot_'+ str(i) + str(training_steps) + '.png')

            if training_steps % num_steps_save_training_run == 0:
                save_training_run(losses[i], rewards[i], agent[i], model_path[i])

    plot_data = list()
    plot_agent = list()

    for i in range(len(agent)):
        writer[i].close()

    for i in range(len(agent)): 

        f =  open('outputs/plot_'+str(i)+'.txt')

        plot_data = list()
        steps = list()
        clock = list()
        rew = list()
        loss = list()
       
        for line in f:
            plot_line = line.replace('\n','').split(" ")

            steps.append(int(plot_line[0]))
            clock.append(float(plot_line[1]))
            rew.append(float(plot_line[2]))
            loss.append(float(plot_line[3]))

            plot_data.append(steps)
            plot_data.append(clock)
            plot_data.append(rew)
            plot_data.append(loss)
        
        plot_agent.append(plot_data)

    plot(plot_agent)
    
# cumulative reward for training and test

def setup_output_dir():
    m_dir = 'outputs/models'
    p_dir = 'outputs/plots'

    if not os.path.exists(m_dir):
        os.makedirs(m_dir)
    if not os.path.exists(p_dir):
        os.makedirs(p_dir)

def main():
    from agent import actions
    state_size = (config2.vision_range*2 + 1)**2 * config2.color_num_dims + config2.scent_num_dims + len(actions)

    agent = list()
    env = list()
    optimizer = list()

    num_agents = 2

    for i in range(num_agents):
        env.append(Environment(config2))
        agent.append(WeightedImitationAgent(env[i], state_size=state_size))

    setup_output_dir()
    train(agent, env, [0, 1, 2, 3])

if __name__ == '__main__':
    main()
