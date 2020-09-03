import numpy as np
import matplotlib.pyplot as plt
import gym


def policy_iter(env, gamma, theta):
    """Policy Iteration Algorithm

    Params:
        env - environment with following required memebers:
        gamma (float) - discount factor
        theta (float) - termination condition
    """

    # 1. Initialization
    state_size = env.env.nS
    action_size = env.env.nA
    V = np.zeros(state_size)
    pi = np.zeros(state_size, dtype=int)  # greedy, always pick action 0

    while True:
        # 2. Policy Evaluation
        while True:
            delta = 0
            for s in range(state_size):
                v = V[s]
                V[s] = sum_sr(env, V=V, s=s, a=pi[s], gamma=gamma)
                delta = max(delta, abs(v - V[s]))
            if delta < theta: break

        # 3. Policy Improvement
        policy_stable = True
        for s in range(state_size):
            old_action = pi[s]
            pi[s] = np.argmax([sum_sr(env, V=V, s=s, a=a, gamma=gamma)
                               for a in range(action_size)])
            if old_action != pi[s]: policy_stable = False
        if policy_stable: break

    return V, pi


def sum_sr(env, V, s, a, gamma):
    """
    Calc state-action value for state 's' and action 'a'

    Args:
        env: the environment
        V: V function
        s: current state
        a: action to take
        gamma: discount factor

    Returns:

    """
    tmp = 0  # state value for state s
    for prob, next_state, reward, _ in env.P[s][a]:  # see note #1 !
        # prob  - transition probability from (s,a) to (s')
        # next_state - next state (s')
        # reward  - reward on transition from (s,a) to (s')
        tmp += prob * (reward + gamma * V[next_state])
    return tmp


if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    env.reset()
    env.render()
    V, pi = policy_iter(env, gamma=1.0, theta=1e-8)
    print(V.reshape([4, -1]))
    print(pi.reshape([4, -1]))
    a2w = {0: '<', 1: 'v', 2: '>', 3: '^'}
    policy_arrows = np.array([a2w[x] for x in pi])
    print(np.array(policy_arrows).reshape([-1, 4]))
