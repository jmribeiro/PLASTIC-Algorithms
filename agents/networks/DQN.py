import numpy as np

from agents.networks.ReplayMemoryModel import ReplayMemoryModel
from yaaf.agents.dqn.networks import DeepQNetwork
from yaaf.models.feature_extraction import MLPFeatureExtractor


class DQN(ReplayMemoryModel):

    """QNetwork for Pursuit"""

    def __init__(self, trainable, learning_rate, discount_factor, layers, replay_batch_size, replay_memory_size):

        super().__init__("dqn", trainable, layers, replay_batch_size, replay_memory_size)

        self.features = (1 + 3) * 2
        self.outputs = 4
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        mlp_feature_extractor = MLPFeatureExtractor(num_inputs=self.features, layers=layers)
        self.model = DeepQNetwork([mlp_feature_extractor], self.outputs, self.learning_rate, cuda=True)

    def predict(self, x):
        q_values = self.model.predict(x.reshape(1, -1))[0]
        return q_values.numpy()

    def fit_batch(self, batch, verbose):
        X, Y = self.prepare_dataset(batch)
        losses, _, accuracies = self.model.fit_and_validate(X, Y, epochs=1, batch_size=self._replay_batch_size, verbose=verbose)
        info = {"loss": losses[-1], "accuracy": accuracies[-1]}
        return info

    def prepare_dataset(self, batch):

        m = len(batch)
        F = self.features
        A = 4

        X = np.zeros((m, F))
        Y = np.zeros((m, A))

        for j, datapoint in enumerate(batch):

            state, joint_actions, reward, next_state, terminal = datapoint

            x = state.features_relative_agent(0)

            if terminal:
                yj = reward
            else:
                next_x = next_state.features_relative_agent(0)
                q_values_next_state = self.predict(next_x)
                yj = (reward + self.discount_factor * q_values_next_state.max())

            # Predict using current network to zero-out the loss on backpropagation.
            Yj = self.predict(x)
            a = joint_actions[0]
            Yj[a] = yj

            X[j] = x
            Y[j] = Yj

        return X, Y

    def save(self, directory):
        super().save(directory)
        self.model.save(f"{directory}/dqn")

    def _load(self, directory):
        self.model.load(f"{directory}/dqn")