from environment.Pursuit import Pursuit


class PerfectEnvironmentModel:

    """ Game code environment model """

    def __init__(self):
        self.name = "Perfect Environment Model"
        self.trainable = False

    def simulate_transition(self, state, actions):
        next_state, reward = Pursuit.transition(state, actions)
        return next_state, reward

    def predict_next_state(self, state, actions):
        return Pursuit.transition(state, actions)[0]

    def predict_reward(self, next_state):
        return Pursuit.reward(next_state)

    def replay_fit(self, datapoint):
        pass

    def save(self, directory):
        pass

    def load(self, directory):
        pass
