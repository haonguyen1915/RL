"""
    training at: https://colab.research.google.com/drive/1xQDvsiQ6Lz7o61nA3aCrruJwYeQH9zu7#scrollTo=Xv9TIS9vilQd
"""
import torch
import gym
from magic.lib_cm import logger
from agent.lunar_lander_agent import LunarLanderAgent
_logger = logger.get_logger(__name__)
env = gym.make('LunarLander-v2')
env.seed(0)
_logger.pr(f'State shape: {env.observation_space.shape}')
_logger.pr(f'Number of actions: {env.action_space.n} ')

state = env.reset()

agent = LunarLanderAgent(state_size=8, action_size=4, seed=0)

# agent.qnetwork_local.load_state_dict(torch.load("models/lunar.pth", map_location=torch.device('cpu')))
for idx in range(10000):
    action = env.action_space.sample()
    action = agent.act(state)
    env.render()
    state, reward, done, _ = env.step(env.action_space.sample())  # take a random action
    _logger.pr(f"action: {action}, reward: {reward}", logger.CYAN)
    if done:
        _logger.pr(f"terminated at: {idx} step")
        break
env.close()
