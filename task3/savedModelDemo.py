import dqn
import gymnasium as gym
import torch
import math
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

targetNet = dqn.DQN(env.observation_space.shape[0], 64, env.action_space.n).to(device)
targetNet.load_state_dict(torch.load('cartPole.pth'))

loops = 100 # How many times you want to watch the model play
for _ in range(loops):
    observation, info = env.reset()
    observation = torch.tensor([observation], device=device, dtype=torch.float)
    thisEpisodes = 0
    while True:
        
        #while we can continue
        action = targetNet(observation).max(1).indices.view(1, 1)
        actionOutcome, reward, terminated, truncated, info = env.step(action.item())

        if terminated or truncated:
            break
        
env.close()