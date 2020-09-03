import gym
import numpy as np
import matplotlib.pyplot as plt


def policy_evaluation(env, policy, gamma, theta):
    """
    Iterative Policy Evaluation

    Args:
        env: environment with following required memebers:
        policy: (2d array) - policy to evaluate, rows must sum to 1
        gamma: discount factor
        theta: termination condition

    Returns:

    """
    action_size = env.env.nA
    state_size = env.env.nS
    V = np.zeros(state_size)

    while True:
        delta = 0
        for s in range(state_size):
            v = V[s]

            sum_prob_trans = 0
            # Calc Sum probability transition to state s' from state s, action a
            # SUM_SIGMA[pi(a, s) * (p(s', r | s, a) * [reward + gamma*V(s')]

            for a in range(action_size):
                for prob, next_state, reward, _ in env.P[s][a]:
                    sum_prob_trans += policy[s, a] * prob * (reward + gamma * V[next_state])
            V[s] = sum_prob_trans

            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V


if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    env.reset()
    policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA
    print(f"policy prob: {policy}")
    print(f"policy shape: {policy.shape}")
    V_pi = policy_evaluation(env, policy, gamma=1.0, theta=1e-8)
    print(f"values table: \n{V_pi.reshape([4, -1])}")
    env.render()

