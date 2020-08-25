from rl.agent.agent import Agent
import gym
from collections import namedtuple, deque
import numpy as np
import torch


class DQNTrainer(object):
    def __init__(self, agent=None, env='CartPole-v0', **kwargs):
        """

        Args:
            agent:
            env: 'CartPole-v0', 'LunarLander-v2'
            **kwargs:
                n_episodes (int): maximum number of training episodes
                max_t (int): maximum number of time-steps per episode
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end (float): minimum value of epsilon
                eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """

        self.env = gym.make(env)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # Init agent
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = Agent(self.state_size, self.action_size, 777)

        self.n_episodes = kwargs.get('n_episodes', 3000)
        self.max_t = kwargs.get('max_t', 1000)
        self.eps_start = kwargs.get('eps_start', 1.0)
        self.eps_end = kwargs.get('eps_end', 0.01)
        self.eps_decay = kwargs.get('eps_decay', 0.995)

    def initialize(self):
        self.env.seed(777)

    def train(self):
        """Deep Q-Learning.

            Params
            ======
                n_episodes (int): maximum number of training episodes
                max_t (int): maximum number of timesteps per episode
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end (float): minimum value of epsilon
                eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start  # initialize epsilon
        best_score = -1
        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(self.max_t):
                action = self.agent.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(self.eps_end, self.eps_decay * eps)  # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)),
                  end="")
            if i_episode % 100 == 0:
                print(
                    '\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) > 180 and np.mean(scores_window) > best_score:
                best_score = np.mean(scores_window)
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode - 100, np.mean(scores_window)))
                torch.save(self.agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            if np.mean(scores_window) >= 200.0:
                break

        return scores
