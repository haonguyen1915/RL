from rl.agent.agent import Agent
from rl.agent.dynamic_programing.q_learning_agent import QLearningAgent
from rl.agent.dynamic_programing.sarsa_agent import SarsaAgent
from rl.agent import *
import gym
from collections import namedtuple, deque
import numpy as np
from rl.enviroments.wrappers import make_atari, wrap_deepmind, wrap_pytorch

import torch
import os
import logging

_logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, agent=None, env='CartPole-v0', **kwargs):
        self.env_name = env
        self.env = gym.make(env)
        if self.env.observation_space.shape:
            self.state_size = self.env.observation_space.shape[0]
        else:
            self.state_size = 1
        self.action_size = self.env.action_space.n

        self.n_episodes = kwargs.get('n_episodes', 3000)
        self.max_t = kwargs.get('max_t', 1000)
        self.eps_start = kwargs.get('eps_start', 1.0)
        self.eps_end = kwargs.get('eps_end', 0.01)
        self.eps_decay = kwargs.get('eps_decay', 0.995)

    def initialize(self):
        self.env.seed(777)

    def train(self):
        pass

    def test(self):
        pass

    def test_env(self):
        for i_episode in range(10):
            state = self.env.reset()
            for t in range(100):
                self.env.render()

                action = self.env.action_space.sample()
                state, reward, done, info = self.env.step(action)
                if t == 0:
                    print(f"state: {self.env.observation_space}")
                    print(f"state shape: {self.env.observation_space.shape}")
                    print(f"num_action: {self.env.action_space.n}")
                    print(f"action: {action}")
                    print(f"reward: {reward}\n")

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        self.env.close()


class QLearningTrainer(Trainer):
    ENVS = ['Taxi-v3']
    def __init__(self, agent=None, env='Taxi-v3', **kwargs):

        kwargs['n_episodes'] = 10000
        if env not in self.ENVS:
            _logger.warning(f"Q-learning only support {self.ENVS}"
                            f"So get defaults env: Taxi-v2")
            env = 'Taxi-v3'

        super(QLearningTrainer, self).__init__(agent, env, **kwargs)
        self.state_size = self.env.observation_space.n
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = QLearningAgent(self.state_size, self.action_size, 777)

    def train(self):
        """

        Returns:

        """
        model_path = f"models/{self.env_name}_{self.agent.__class__.__name__}_model.pth"
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start  # initialize epsilon
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
                print(
                    '\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)),
                    end="")
                if i_episode % 100 == 0:
                    print(
                        '\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode,
                                                                     np.mean(scores_window)))
        self.agent.save(model_path)

    def test(self):
        model_path = f"models/{self.env_name}_{self.agent.__class__.__name__}_model.pth"
        self.agent.load(model_path)
        state = self.env.reset()
        for idx in range(10000):
            # action = env.action_space.sample()
            action = self.agent.act(state)
            self.env.render()
            state, reward, done, _ = self.env.step(action)  # take a random action
            print(f"action: {action}, reward: {reward}")
            if done:
                print(f"terminated at: {idx} step")
                break
        self.env.close()


class SarsaTrainer(Trainer):
    ENVS = 'Taxi-v3'

    def __init__(self, agent=None, env='Taxi-v2', **kwargs):

        kwargs['n_episodes'] = 10000
        if env not in self.ENVS:
            _logger.warning(f"Sarsa only support {self.ENVS}"
                            f"So get defaults env: Taxi-v2")
            env = 'Taxi-v3'

        super(SarsaTrainer, self).__init__(agent, env, **kwargs)
        self.state_size = self.env.observation_space.n
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = SarsaAgent(self.state_size, self.action_size, 777)

    def train(self):
        """

        Returns:

        """
        model_path = f"models/{self.env_name}_{self.agent.__class__.__name__}_model.pth"
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start  # initialize epsilon
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
                print(
                    '\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)),
                    end="")
                if i_episode % 100 == 0:
                    print(
                        '\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode,
                                                                     np.mean(scores_window)))
        self.agent.save(model_path)

    def test(self):
        model_path = f"models/{self.env_name}_{self.agent.__class__.__name__}_model.pth"
        self.agent.load(model_path)
        state = self.env.reset()
        for idx in range(10000):
            # action = env.action_space.sample()
            action = self.agent.act(state)
            self.env.render()
            state, reward, done, _ = self.env.step(action)  # take a random action
            print(f"action: {action}, reward: {reward}")
            if done:
                print(f"terminated at: {idx} step")
                break
        self.env.close()


class DQNTrainer(Trainer):
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
        super(DQNTrainer, self).__init__(agent, env, **kwargs)
        # Init agent
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = DQNAgent(self.state_size, self.action_size)

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
        model_path = f"models/{self.env_name}_{self.agent.__class__.__name__}_model.pth"
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start  # initialize epsilon
        best_score = -1
        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(self.max_t):
                action = self.agent.act(state, eps)
                # action = self.agent.main_network.act(state)
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
                self.agent.save(model_path)
            if np.mean(scores_window) >= 200.0:
                break

        return scores

    def test(self):
        model_path = f"models/{self.env_name}_{self.agent.__class__.__name__}_model.pth"
        self.agent.load(model_path)
        state = self.env.reset()

        for idx in range(10000):
            # action = env.action_space.sample()
            action = self.agent.act(state)
            self.env.render()
            state, reward, done, _ = self.env.step(action)  # take a random action
            print(f"action: {action}, reward: {reward}")
            if done:
                print(f"terminated at: {idx} step")
                break
        self.env.close()


class DoubleDQNTrainer(DQNTrainer):
    def __init__(self, agent=None, env='CartPole-v0', **kwargs):
        super(DoubleDQNTrainer, self).__init__(agent, env, **kwargs)
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = DoubleDQNAgent(self.state_size, self.action_size, 777)


class PerDQNTrainer(DQNTrainer):
    def __init__(self, agent=None, env='CartPole-v0', **kwargs):
        super(PerDQNTrainer, self).__init__(agent, env, **kwargs)
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = PerDQNAgent(self.state_size, self.action_size, 777)


class DualDQNTrainer(DQNTrainer):
    def __init__(self, agent=None, env='CartPole-v0', **kwargs):
        super(DualDQNTrainer, self).__init__(agent, env, **kwargs)
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = DualDQNAgent(self.state_size, self.action_size, 777)


class DualDoubleDQNTrainer(DualDQNTrainer):
    def __init__(self, agent=None, env='CartPole-v0', **kwargs):
        super(DualDoubleDQNTrainer, self).__init__(agent, env, **kwargs)


class NoiseDQNTrainer(DQNTrainer):
    def __init__(self, agent=None, env='CartPole-v0', **kwargs):
        super(NoiseDQNTrainer, self).__init__(agent, env, **kwargs)
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = NoiseDQNAgent(self.state_size, self.action_size, 777)
        self.n_episodes = 10000


class DistPerspectiveDQNTrainer(DQNTrainer):
    def __init__(self, agent=None, env='CartPole-v0', **kwargs):
        super(DistPerspectiveDQNTrainer, self).__init__(agent, env, **kwargs)
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = DistPerspectiveDQNAgent(self.state_size, self.action_size)
        self.n_episodes = 10000


class RainBowTrainer(DQNTrainer):
    def __init__(self, agent=None, env='CartPole-v0', **kwargs):
        super(RainBowTrainer, self).__init__(agent, env, **kwargs)
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = RainBowAgent(self.state_size, self.action_size)
        self.n_episodes = 15000


class DQNCNNTrainer(Trainer):
    ENVS = ['PongNoFrameskip-v4']

    def __init__(self, agent=None, env='PongNoFrameskip-v4', **kwargs):
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
        if env not in self.ENVS:
            _logger.warning(f"CnnDQN only support {self.ENVS}"
                            f"So get defaults env: PongNoFrameskip-v4")
            env = 'PongNoFrameskip-v4'
        super(DQNCNNTrainer, self).__init__(agent, env, **kwargs)
        self.env = make_atari(env)
        self.env = wrap_deepmind(self.env)
        self.env = wrap_pytorch(self.env)
        # Init agent
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = CnnDQNAgent(self.env.observation_space.shape, self.action_size, 777)

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
        model_path = f"models/{self.env_name}_{self.agent.__class__.__name__}_model.pth"
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
                self.agent.save(model_path)
            if np.mean(scores_window) >= 200.0:
                break

        return scores

    def test(self):
        model_path = f"models/{self.env_name}_{self.agent.__class__.__name__}_model.pth"
        self.agent.load(model_path)
        state = self.env.reset()

        for idx in range(10000):
            # action = env.action_space.sample()
            action = self.agent.act(state)
            self.env.render()
            state, reward, done, _ = self.env.step(action)  # take a random action
            print(f"action: {action}, reward: {reward}")
            if done:
                print(f"terminated at: {idx} step")
                break
        self.env.close()

    def test_env(self):
        """

        Returns:

        """
        for i_episode in range(10):
            state = self.env.reset()
            for t in range(100):
                self.env.render()

                action = self.env.action_space.sample()
                state, reward, done, info = self.env.step(action)
                if t == 0:
                    print(f"state: {self.env.observation_space.shape}")
                    print(f"num_action: {self.env.action_space.n}")
                    print(f"action: {action}")
                    print(f"reward: {reward}\n")

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        self.env.close()
