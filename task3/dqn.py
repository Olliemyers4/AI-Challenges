import torch
from collections import namedtuple, deque
import random


class DQN(torch.nn.Module):
    def __init__(self, inputSize,hiddenSize, outputSize):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize)
        self.fc2 = torch.nn.Linear(hiddenSize, hiddenSize)
        self.fc3 = torch.nn.Linear(hiddenSize, outputSize)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Transition(object):
    def __init__(self, state, action, reward, nextState):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
    
class ReplayMemory(object):
    def __init__(self, capacity):
        self.mem = deque(maxlen=capacity)
    
    def push(self, transition):
        self.mem.append(transition)
    
    def sample(self, batchSize):
        return random.sample(self.mem, batchSize)
    
    def __len__(self):
        return len(self.mem)