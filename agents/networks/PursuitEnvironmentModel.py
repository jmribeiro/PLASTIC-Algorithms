import random
from abc import abstractmethod, ABC

from agents.networks.FeedForwardNetwork import FeedForwardNetwork
from agents.networks.ReplayMemoryModel import ReplayMemoryModel
from agents.networks.utils import remove_collisions
import numpy as np


class PursuitEnvironmentModel(ReplayMemoryModel, ABC):

    def __init__(self, num_agents, name, trainable, learning_rate, layers, replay_batch_size, replay_memory_size, fix_collisions, learn_reward):

        super().__init__(name, trainable, layers, replay_batch_size, replay_memory_size)

        self.fix_collisions = fix_collisions
        self.num_agents = num_agents
        if learn_reward:
            self.rewards_model = FeedForwardNetwork(num_inputs=num_agents * 2, num_outputs=2, layers=((64, "relu"),), learning_rate=0.01, cuda=True)
            self.terminal_states = []
            self.non_terminal_states = []

    def remember(self, pursuit_datapoint):
        _, _, _, next_state, terminal = pursuit_datapoint
        if hasattr(self, "rewards_model"):
            if terminal: self.terminal_states.append(next_state.features_relative_prey())
            else: self.non_terminal_states.append(next_state.features_relative_prey())
        super().remember(pursuit_datapoint)

    def simulate_transition(self, state, actions):
        """ Used by MCTS """
        next_state = self.predict_next_state(state, actions)
        reward = self.predict_reward(next_state)
        return next_state, reward

    def predict_next_state(self, state, actions):
        """ Predicts a next state given a previous state and the joint actions
            Corrects the next_state if invalid for the MDP.
        """
        next_possible_state = self._predict_next_state(state, actions)
        if self.fix_collisions: next_state = remove_collisions(state, next_possible_state)
        else: next_state = next_possible_state
        return next_state

    def predict_reward(self, next_state):
        """Predicts the reward rt for a given state st+1"""
        if hasattr(self, "rewards_model"):
            features = next_state.features_relative_prey()
            x = features.reshape(1, -1)
            y = self.rewards_model.classify(x)
            is_terminal = y == 1
            return 100 if is_terminal else -1.0  # This is not necessary - Consider changing to "terminal predictor"
        else:
            return 100 if next_state.is_terminal else -1.0

    def fit_batch(self, batch, verbose=False):
        info = {}
        if hasattr(self, "rewards_model"):
            info["reward net"] = self.update_reward_predictor(verbose)
        info["transition net"] = self.update_state_predictors(batch, verbose)
        return info

    def update_reward_predictor(self, verbose):
        try:
            X, y = self.balance_rewards_dataset(self.terminal_states, self.non_terminal_states, self._replay_batch_size)
            losses, _, accuracies = self.rewards_model.fit_and_validate(X, y, epochs=1, batch_size=self._replay_batch_size, verbose=verbose)
            info = {"loss": losses[-1], "accuracy": accuracies[-1], "samples": len(self.terminal_states)}
        except ValueError:
            info = {}
        return info

    @abstractmethod
    def update_state_predictors(self, epochs, verbose):
        raise NotImplementedError()

    @staticmethod
    def balance_rewards_dataset(terminal_states, non_terminal_states, min_batch=32):

        num_terminal_states = len(terminal_states)
        num_non_terminal_states = len(non_terminal_states)

        if num_terminal_states > 0:

            # Take at much as half of the batches
            half = int(min_batch/2)
            num_terminal_samples = min(half, num_terminal_states)
            Xterminal = random.sample(population=terminal_states, k=num_terminal_samples)

            # If the number of terminal states is lower than half, take equally from non terminal states
            num_non_terminal_samples = min(num_terminal_samples, num_non_terminal_states)
            Xnonterminal = random.sample(population=non_terminal_states, k=num_non_terminal_samples)

            y = [1 for _ in range(len(Xterminal))]
            y.extend([0 for _ in range(len(Xnonterminal))])
            X = np.array(Xterminal + Xnonterminal)
            y = np.array(y)
            return X, y

        else:
            raise ValueError("No terminal states")

    @staticmethod
    def create_rewards_dataset(dataset):
        X_terminal = []
        X_non_terminal = []
        for i, (_, _, _, next_state, _) in enumerate(dataset):
            if next_state.is_terminal:
                X_terminal.append(next_state.features_relative_prey())
            else:
                X_non_terminal.append(next_state.features_relative_prey())
        return PursuitEnvironmentModel.balance_rewards_dataset(X_terminal, X_non_terminal, min_batch=32)

    @abstractmethod
    def _predict_next_state(self, state, actions):
        raise NotImplementedError()

    def save(self, directory):
        super().save(directory)
        if hasattr(self, "rewards_model"):
            self.rewards_model.save(f"{directory}/rewards")
        self._save(directory)

    def _save(self, directory):
        raise NotImplementedError()

    def _load(self, directory):
        if hasattr(self, "rewards_model"):
            self.rewards_model.load(f"{directory}/rewards")
