import numpy as np
import matplotlib.pyplot as plt
import gym


def value_iter(env, gamma, theta):
    """Value Iteration

    Params:
        env - environment with following required memebers:
        gamma (float) - discount factor
        theta (float) - termination condition
    """
    state_size = env.env.nS
    action_size = env.env.nA
    V = np.zeros(state_size)

    while True:
        delta = 0
        for s in range(state_size):
            v = V[s]
            V[s] = max([sum_sr(env, V=V, s=s, a=a, gamma=gamma)
                        for a in range(action_size)])
            delta = max(delta, abs(v - V[s]))

        if delta < theta: break
    # Output a deterministic policy
    pi = np.zeros(state_size, dtype=int)
    for s in range(state_size):
        pi[s] = np.argmax([sum_sr(env, V=V, s=s, a=a, gamma=gamma)  # list comprehension
                           for a in range(action_size)])

    return V, pi


def sum_sr(env, V, s, a, gamma):
    """
    Calc state-action value for state 's' and action 'a'

    Args:
        env:
        V:
        s:
        a:
        gamma:

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
    V, pi = value_iter(env, gamma=1.0, theta=1e-8)
    print(V.reshape([4, -1]))
    a2w = {0: '<', 1: 'v', 2: '>', 3: '^'}
    policy_arrows_origin = np.array([a2w[x] for x in pi])
    policy_arrows = np.array([x for x in pi])
    print(np.array(policy_arrows_origin).reshape([-1, 4]))
    print(np.array(policy_arrows).reshape([-1, 4]))
