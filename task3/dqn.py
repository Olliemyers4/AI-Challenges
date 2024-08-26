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
    
Transition = namedtuple('Transition', ('observation', 'action', 'nextObservation', 'reward'))
    
class ReplayMemory(object):
    def __init__(self, capacity):
        self.mem = deque([],maxlen=capacity)
    
    def push(self, *transition):
        self.mem.append(Transition(*transition))
    
    def sample(self, batchSize):
        return random.sample(self.mem, batchSize)
    
    def __len__(self):
        return len(self.mem)
    
class Parameters(object):
    def __init__(self, batchSize, gamma,device,optimiser,learningRate,tau):
        self.batchSize = batchSize
        self.gamma = gamma
        self.device = device
        self.optimiser = optimiser
        self.learningRate = learningRate
        self.tau = tau


def selectAction(observation,epsilonThreshold,env,model,params):
    # epsilon greedy
    epsilonRandom = random.random()
    if epsilonRandom < epsilonThreshold:
        # greedy
        return torch.tensor([[env.action_space.sample()]], device=params.device, dtype=torch.long)
    else:
        # what the network thinks
        with torch.no_grad():
            return model(observation).max(1).indices.view(1, 1)


def optimiser(params,memory,policyModel,targetModel):
    if len(memory) < params.batchSize:
        return # not enough samples to perform sampling
    transitions = memory.sample(params.batchSize)
    batchedSamples = Transition(*zip(*transitions))

    # filter out the final states - boolean mask
    nonFinalBoolMask = torch.tensor(tuple(map(lambda s: s is not None, batchedSamples.nextObservation)), device=params.device, dtype=torch.bool)

    # filter out the final states
    nonFinalNextObservations= torch.cat([s for s in batchedSamples.nextObservation if s is not None])

    # Create the batch tensors - for the observations, actions and rewards
    # This is needed to calculate the expected observation action values
    # to then calculate the loss
    observationBatch = torch.cat(batchedSamples.observation)
    actionBatch = torch.cat(batchedSamples.action)
    rewardBatch = torch.cat(batchedSamples.reward)

    # Run these through the network
    ActionValues = policyModel(observationBatch).gather(1, actionBatch)
    nextActionValues = torch.zeros(params.batchSize, device=params.device)
    with torch.no_grad():
        nextActionValues[nonFinalBoolMask] = targetModel(nonFinalNextObservations).max(1).values
  
    expectedActionValues = rewardBatch + (nextActionValues * params.gamma)

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(ActionValues, expectedActionValues.unsqueeze(1))

 
    params.optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policyModel.parameters(),1)
    params.optimiser.step()
    
