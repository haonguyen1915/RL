from collections import defaultdict
import torch
import gym
import numpy as np
import random

gamma = 0.9
n_episode = 10000
optimal_policy = torch.tensor([0., 3., 3., 3.,
                               0., 3., 2., 3.,
                               3., 1., 0., 3.,
                               3., 2., 1., 3.])


def argmax(numpy_array):
    """ argmax implementation that chooses randomly between ties """
    max_indices = np.where(numpy_array == numpy_array.max())[0]
    return max_indices[np.random.randint(max_indices.shape[0])]


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def run_episode(env, policy, S0, A0):
    state = S0
    states = []
    actions = []
    rewards = []
    next_states = []
    is_done = False
    first_time = True
    while not is_done:
        if first_time:
            env.reset()
            action = A0
            first_time = False
        else:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
        next_state, reward, is_done, info = env.step(action)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        next_states.append(next_state)
        state = next_state
        if is_done:
            break
    return states, actions, rewards, next_states


def monte_carlo_ES_control(env, gamma, n_episode):
    epsilon = 0.1
    pi = defaultdict(int)  # default action: 0
    # Q = defaultdict(float)  # default Q value: 0
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.env.nA)

    Returns = defaultdict(list)  # dict of lists
    for episode in range(n_episode):
        # Randomly select S0, A0
        S0 = random.randint(0, env.env.nS - 1)
        A0 = random.randint(0, env.env.nA - 1)
        states, actions, rewards, next_states = run_episode(env, policy, S0, A0)
        return_t = 0
        T = len(states)
        for t in range(T - 1, -1, -1):
            state_t = states[t]
            reward_t = rewards[t]
            action_t = actions[t]
            return_t = gamma * return_t + reward_t
            if (state_t, action_t) not in [(states[i], actions[i]) for i in range(0, t)]:
                Returns[(state_t, action_t)].append(return_t)
                Q[(state_t, action_t)] = np.average(Returns[(state_t, action_t)])
                # pi[state_t] = np.argmax([Q[state_t, a] for a in range(env.env.nA)])

    return Q, pi


if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    env.render()
    Q, pi = monte_carlo_ES_control(env, gamma, n_episode)
    print(Q)
    # print(pi)
    # print('The value function calculated by first-visit MC prediction:\n', value.reshape([4, -1]))
