import numpy as np
from collections import deque

import environment as env
import nel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from six.moves import cPickle
from torch.autograd import Variable

from random import Random
from datetime import datetime

actions = [nel.Direction.UP, nel.Direction.DOWN, nel.Direction.LEFT,
           nel.Direction.RIGHT]
torch.set_printoptions(precision=10)


class Policy(nn.Module):
    def __init__(self, state_size, action_dim=len(actions), history_len=2,
                 hidden_size=128):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size * history_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)

class BaseAgent(nel.Agent):
    def __init__(self, env, load_filepath=None):
        super(BaseAgent, self).__init__(env.simulator, load_filepath)
        self.env = env

    def save(self):
        pass

    def _load(self):
        pass

    def next_move(self):
        pass

    def step(self):
        pass

class ImitationAgent(BaseAgent):
    def __init__(self, env, state_size, history_len=1, load_filepath=None):
        super(RLAgent, self).__init__(env, load_filepath)
        self.policy = Policy(state_size=state_size)
        self.target = Policy(state_size=state_size)
        self.target.load_state_dict(self.policy.state_dict())
        self.prev = torch.Tensor([0, 0, 0, 0])
        self.prev_action = np.zeros(len(actions), dtype=np.float32)

        for param in self.target.parameters():
            param.requires_grad = False
        self.prev_states = deque(maxlen=history_len)
        self.history_len = history_len

        # Multi-Agent Customization
        self.addOwnExperienceProbability = 0.30
        self.jellybeanAccuracyForAllAgents = [] # contains tuples, (num jellybeans collected, numSteps).
        self.weightsForAllAgents = []
        self.selfAgentID = None
        self.randomGenerator = Random(datetime.now())

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def move_fn(self, epsilon):
        return lambda epsilon=epsilon: self.next_move(epsilon=epsilon)

    def next_move(self, epsilon=0.05):
        # If the current sequence of states is less than the allowed min history
        # length, we randomly pick an action and add it to the history.
        if len(self.prev_states) < self.history_len:
            self.prev_states.append(self.create_current_frame())
            # return actions[np.random.randint(0, len(actions))]
            return np.random.choice(actions) #p=[0.5, 0.1, 0.2, 0.2])

        # If we have a full history, with probability epsilon, we pick a random
        # action, in order to explore.
        random_prob = np.random.rand()
        if random_prob < epsilon:
            self.prev_states.append(self.create_current_frame())
            return np.random.choice(actions) #p=[0.5, 0.1, 0.2, 0.2])
            # return actions[np.random.randint(0, len(actions))]

        # If we don't explore, then we chose the action with max Q value.
        state = self.get_state()
        context = Variable(torch.from_numpy(state), requires_grad=False)
        self.prev_states.append(self.create_current_frame())
        qs = self.policy(context)
        self.prev = qs.data
        # Pick the argmax.
        ind = np.argmax(qs.data.numpy())
        # Alternatively, you can sample proportional to qs.
        return actions[ind]

    def create_current_frame(self):
        vis = self.vision().flatten()
        smell = self.scent()
        return np.concatenate([vis, smell, self.prev_action])

    def get_state(self):
        if len(self.prev_states) > 0:
            context = np.concatenate(self.prev_states)
        else:
            context = np.array([])
        return np.concatenate([context, self.create_current_frame()])

    def step(self, epsilon=0.05):
        current_step_and_reward = self.env.step(self, self.move_fn(epsilon))
        self.prev_action = np.zeros(len(actions), dtype=np.float32)
        self.prev_action[current_step_and_reward[0].value] = 1.0
        return current_step_and_reward

    def save(self, filepath):
        target_path = filepath+'.target'
        with open(target_path, 'wb') as f:
            torch.save(self.target, f)

        model_path = filepath+'.model'
        with open(model_path, 'wb') as f:
            torch.save(self.policy, f)

    def _load(self, filepath):
        target_path = filepath+'.target'
        with open(target_path, 'rb') as f:
            self.target = torch.load(f)

        model_path = filepath+'.model'
        with open(model_path, 'rb') as f:
            self.policy = torch.load(f)

    # Takes in a list of of all agent's experiences, including this one. This
    # agent then updates its weights about which agents to follow, as well as
    # returns which experience it wants to push into its experience replay
    # buffer. Experience is of the form:
    # (state1, action, reward, state2, False).
    # selfI is the index in the list corresponding to this agent's experience.
    # Note that the agents need to be in the same order each time, so that
    # the agent can maintain accurate weights for all oher agents.
    def receiveAllAgentsExperience(self, allAgentsExp, selfI):
        # Check that the agentID has not changed
        if self.selfAgentID is None:
            self.selfAgentID = int(selfI)
        elif self.selfAgentID != int(selfI):
            print("ERROR selfI has changed from %d to %d" % (self.selfAgentID, selfI))
        # If the number of agents has grown (new agents are appended to the end
        # of the list), extend the data structures for keeping track of all agents
        numPrevAgents = len(self.jellybeanAccuracyForAllAgents)
        numCurrAgents = len(allAgentsExp)
        while numCurrAgents > numPrevAgents: # agents can only get added, never die
            self.jellybeanAccuracyForAllAgents.append((0, 1)) # All agents start with 1 step to avoid divide by 0 errors
            self.weightsForAllAgents.append(0.0)
            numPrevAgents++
        # Update agent jellybeanAccuracy and weights
        for i in xrange(len(allAgentsExp)):
            (s1, action, reward, s2, done) = allAgentsExp[i]
            (prevJellybeans, prevSteps) = self.jellybeanAccuracyForAllAgents[i]
            if reward > 0: # If the agent got a jellybean
                self.jellybeanAccuracyForAllAgents[i] = (prevJellybeans + 1, prevSteps + 1)
            else:
                self.jellybeanAccuracyForAllAgents[i] = (prevJellybeans, prevSteps + 1)
            self.weightsForAllAgents[i] = self.weightFromTuple(jellybeanAccuracyForAllAgents[i])
        # Get Probabilities For Picking An Agent
        sumOfWeights = sum(self.weightsForAllAgents[:selfI] + self.weightsForAllAgents[selfI+1:])
        probabilities = [float(weight) * (1 - self.addOwnExperienceProbability) / sumOfWeights for weight in self.weightsForAllAgents]
        probabilities[selfI] = self.addOwnExperienceProbability
        print("Sanity Check: Sum of Probabilies = "+str(sum(probabilities)))
        # Determine which agent's experience to add into the replay buffer
        randomFloat = self.randomGenerator.random()
        totalProbailitySoFar = 0
        for i in xrange(len(probabilities)):
            totalProbailitySoFar += probabilities[i]
            if i == len(probabilities) - 1:
                totalProbailitySoFar = 1
            if randomFloat <= totalProbailitySoFar:
                return allAgentsExp[i]

    def weightFromTuple(self, tup):
        return float(tup[0])/tup[1]

class RLAgent(BaseAgent):
    def __init__(self, env, state_size, history_len=1, load_filepath=None):
        super(RLAgent, self).__init__(env, load_filepath)
        self.policy = Policy(state_size=state_size)
        self.target = Policy(state_size=state_size)
        self.target.load_state_dict(self.policy.state_dict())
        self.prev = torch.Tensor([0, 0, 0, 0])
        self.prev_action = np.zeros(len(actions), dtype=np.float32)

        for param in self.target.parameters():
            param.requires_grad = False
        self.prev_states = deque(maxlen=history_len)
        self.history_len = history_len

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def move_fn(self, epsilon):
        return lambda epsilon=epsilon: self.next_move(epsilon=epsilon)

    def next_move(self, epsilon=0.05):
        # If the current sequence of states is less than the allowed min history
        # length, we randomly pick an action and add it to the history.
        if len(self.prev_states) < self.history_len:
            self.prev_states.append(self.create_current_frame())
            # return actions[np.random.randint(0, len(actions))]
            return np.random.choice(actions) #p=[0.5, 0.1, 0.2, 0.2])

        # If we have a full history, with probability epsilon, we pick a random
        # action, in order to explore.
        random_prob = np.random.rand()
        if random_prob < epsilon:
            self.prev_states.append(self.create_current_frame())
            return np.random.choice(actions) #p=[0.5, 0.1, 0.2, 0.2])
            # return actions[np.random.randint(0, len(actions))]

        # If we don't explore, then we chose the action with max Q value.
        state = self.get_state()
        context = Variable(torch.from_numpy(state), requires_grad=False)
        self.prev_states.append(self.create_current_frame())
        qs = self.policy(context)
        self.prev = qs.data
        # Pick the argmax.
        ind = np.argmax(qs.data.numpy())
        # Alternatively, you can sample proportional to qs.
        return actions[ind]

    def create_current_frame(self):
        vis = self.vision().flatten()
        smell = self.scent()
        return np.concatenate([vis, smell, self.prev_action])

    def get_state(self):
        if len(self.prev_states) > 0:
            context = np.concatenate(self.prev_states)
        else:
            context = np.array([])
        return np.concatenate([context, self.create_current_frame()])

    def step(self, epsilon=0.05):
        current_step_and_reward = self.env.step(self, self.move_fn(epsilon))
        self.prev_action = np.zeros(len(actions), dtype=np.float32)
        self.prev_action[current_step_and_reward[0].value] = 1.0
        return current_step_and_reward

    def save(self, filepath):
        target_path = filepath+'.target'
        with open(target_path, 'wb') as f:
            torch.save(self.target, f)

        model_path = filepath+'.model'
        with open(model_path, 'wb') as f:
            torch.save(self.policy, f)

    def _load(self, filepath):
        target_path = filepath+'.target'
        with open(target_path, 'rb') as f:
            self.target = torch.load(f)

        model_path = filepath+'.model'
        with open(model_path, 'rb') as f:
            self.policy = torch.load(f)



class RandomAgent(BaseAgent):
    def __init__(self, env, load_filepath=None):
        super(RandomAgent, self).__init__(env, load_filepath)

    def next_move(self):
        return np.random.choice(actions, p=[0.5, 0.1, 0.2, 0.2])

    def step(self):
        return self.env.step(self, self.next_move)

    def save(self, filepath):
        pass

    def _load(self, filepath):
        pass


if __name__ == '__main__':
    from config import *

    env0 = env.Environment(config1)
    agent = RLAgent(env0, 30)
