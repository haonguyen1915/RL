from rl.agent import Agent
import numpy as np
import random
import pickle


class QLearningAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """
        Initialize an Agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        super(QLearningAgent, self).__init__(state_size, action_size)
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros([state_size, action_size])
        self.alpha = 0.5
        self.gamma = 0.6

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        experiences = state, action, reward, next_state, done
        self.train_step(experiences, self.gamma)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        Returns:

        """

        if random.random() > eps:
            action = np.argmax(self.q_table[state])  # Exploit learned values
        else:
            action = random.choice(np.arange(self.action_size))  # Explore action space
        return action

    def train_step(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Following bellow equation
            Q(S, A) <- Q(S, A) + alpha[R + gamma * maxQ(S', A') - Q(S, A)
            =>  Q(S, A) <- (1-alpha) * Q(S, A) + alpha[R + gamma * maxQ(S', A')
                S <- S'
        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor

        Returns:

        """

        state, action, reward, next_state, done = experiences
        old_value = self.q_table[state, action]
        next_value_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + gamma * next_value_max)
        self.q_table[state, action] = new_value

    def save(self, model_path):
        """

        Returns:

        """
        data = {
            'q_table': self.q_table
        }
        with open(model_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved q-table in to {model_path}")

    def load(self, model_path):
        """

        Returns:

        """
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        print(f"Loaded q-table from {model_path}")
