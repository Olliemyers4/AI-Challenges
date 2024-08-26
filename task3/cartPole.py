import dqn
import gymnasium as gym
import torch
import math

def epsilonDecay(start,end,decayRate,steps):
    return end + (start - end) * math.exp(-1. * steps / decayRate)

env = gym.make('CartPole-v1',render_mode = "human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policyNet = dqn.DQN(env.observation_space.shape[0], 32, env.action_space.n).to(device)
targetNet = dqn.DQN(env.observation_space.shape[0], 32, env.action_space.n).to(device)
targetNet.load_state_dict(policyNet.state_dict())

memory = dqn.ReplayMemory(1000)
batchSize = 32
episodes = 1000
gamma = 0.99 # discount factor
lr = 0.001
optimiser = torch.optim.Adam(policyNet.parameters(), lr=lr)
tau = 0.01 # soft update rate
params = dqn.Parameters(batchSize=batchSize,gamma=gamma,device=device,optimiser=optimiser,learningRate=lr,tau=tau)
steps = 0

epsStart = 1
epsEnd = 0.01
epsDecay = 1000


for _ in range(episodes):
    observation, info = env.reset()
    observation = torch.tensor([observation], device=params.device, dtype=torch.float)

    while True:
        #while we can continue
        epsilonThreshold = epsilonDecay(epsStart,epsEnd,epsDecay,steps)
        action = dqn.selectAction(observation=observation,epsilonThreshold=epsilonThreshold,env=env,model=policyNet,params=params)
        steps += 1
        actionOutcome, reward, terminated, truncated, info = env.step(action.item())

        reward = torch.tensor([reward], device=params.device)
        
        if (terminated):
            nextObservation = None
        else:
            nextObservation = torch.tensor([actionOutcome], device=params.device, dtype=torch.float)

        # Add to the transition memory
        memory.push(observation, action,nextObservation,reward)
      
        observation = nextObservation

        dqn.optimiser(params=params,memory=memory,policyModel=policyNet,targetModel=targetNet)

        targetState = targetNet.state_dict()
        policyState = policyNet.state_dict()
        for key in policyState:
            targetState[key] = policyState[key] * params.tau + targetState[key] * (1.0 - params.tau)

        if (terminated or truncated):
            break

env.close()