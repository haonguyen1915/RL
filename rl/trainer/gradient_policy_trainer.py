from .trainer import Trainer
from rl.agent.reinforce_agent import ReinforceAgent, A2CAgent
from collections import deque
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma ** i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def compute_returns(next_value, rewards, masks=None, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns


class GradientPolicyTrainer(Trainer):
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
        super(GradientPolicyTrainer, self).__init__(agent, env, **kwargs)
        # Init agent
        self.n_episodes = 10000
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = ReinforceAgent(self.state_size, self.action_size)

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
            states = []
            rewards = []
            actions = []
            next_states = []
            dones = []
            complete = False
            for t in range(self.max_t):
                action = self.agent.act(state, eps)
                # action = self.agent.main_network.act(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                rewards.append(reward)
                actions.append(action)
                next_states.append(next_state)
                dones.append(done)
                state = next_state

                # self.agent.step(state, action, reward, next_state, done)
                #
                # state = next_state
                score += reward
                if done:
                    # rewards = discount_rewards(rewards, 0.99)
                    self.agent.batch_step(states, actions, rewards, next_states, dones)

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


class A2CTrainer(GradientPolicyTrainer):
    def __init__(self, agent=None, env='CartPole-v0', **kwargs):
        super(A2CTrainer, self).__init__(agent, env, **kwargs)
        self.n_episodes = 10000
        if agent is not None:
            raise NotImplementedError
        else:
            self.agent = A2CAgent(self.state_size, self.action_size)

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
        entropy_term = 0
        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            log_probs = []
            score = 0
            states = []
            rewards = []
            actions = []
            next_states = []
            values = []
            dones = []
            for t in range(self.max_t):
                state = torch.FloatTensor(state).to(device)

                value, policy_dist = self.agent.network(state.unsqueeze(0))

                # value = value.detach().numpy()[0, 0]
                dist = policy_dist.detach().numpy()

                action = np.random.choice(self.agent.action_size, p=np.squeeze(dist))
                # log_prob = torch.log(policy_dist.squeeze(0)[action])
                # entropy = -np.sum(np.mean(dist) * np.log(dist))

                # action = self.agent.main_network.act(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                rewards.append(reward)
                actions.append(action)
                next_states.append(next_state)
                dones.append(done)
                # values.append(value)
                # log_probs.append(log_prob)
                # entropy_term += entropy

                state = next_state
                score += reward
                if done or t == self.max_t - 1:
                    self.agent.batch_step(states, actions, rewards, next_states, dones)
                    break

            # next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            # next_value, _ = self.agent.network(next_state)
            # next_value = next_value.detach().numpy()[0, 0]
            #
            # returns = np.zeros_like(values)
            # for t in reversed(range(len(rewards))):
            #     next_value = rewards[t] + 0.99 * next_value
            #     returns[t] = next_value
            # # update actor critic
            # values = torch.FloatTensor(values)
            # returns = torch.FloatTensor(returns)
            # log_probs = torch.stack(log_probs)

            # print(f"rewards: {rewards}")
            # print(f"log_probs: {log_probs.size()}")
            # print(f"next_state: {next_state.size()}")
            # print(f"values: {values.size()}")
            # print(f"returns: {returns.size()}")
            # print(f"returns: {returns}")
            # exit()

            # advantage = returns - values
            # actor_loss = (-log_probs * advantage).mean()
            # critic_loss = 0.5 * advantage.pow(2).mean()
            # ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
            #
            # self.agent.optimizer.zero_grad()
            # ac_loss.backward()
            # self.agent.optimizer.step()

            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(self.eps_end, self.eps_decay * eps)  # decrease epsilon
            # print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score),
            #       end="")
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
