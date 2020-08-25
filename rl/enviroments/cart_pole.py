"""
    training at: https://colab.research.google.com/drive/1pNYWPX6f_OL11mKz_6WugNf4KtpmllU9#scrollTo=a3XqB4gAqXYO
"""
import torch
import gym
from magic.lib_cm import logger
from agent.agent import Agent

_logger = logger.get_logger(__name__)
env = gym.make('CartPole-v0')
env.seed(0)
_logger.pr(f'State shape: {env.observation_space.shape}')
_logger.pr(f'Number of actions: {env.action_space.n} ')

state = env.reset()

agent = Agent(state_size=4, action_size=2, seed=0)

agent.qnetwork_local.load_state_dict(
    torch.load("models/cart_pole.pth", map_location=torch.device('cpu')))
# agent.qnetwork_target.load_state_dict(
#     torch.load("models/cart_pole.pth", map_location=torch.device('cpu')))

# print(agent.qnetwork_target.fc2.weight)
# exit()
for idx in range(10000):
    # action = env.action_space.sample()
    action = agent.act(state)
    env.render()
    state, reward, done, _ = env.step(action)  # take a random action
    _logger.pr(f"action: {action}, reward: {reward}", logger.CYAN)
    if done:
        _logger.pr(f"terminated at: {idx} step")
        break
env.close()
