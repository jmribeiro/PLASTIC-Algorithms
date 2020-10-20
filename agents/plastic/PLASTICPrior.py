from abc import ABC, abstractmethod

import numpy as np


class PLASTICPrior(ABC):

    """ Encapsulates a team model. """

    def __init__(self, name, num_teammates):
        self.name = name
        self.num_teammates = num_teammates
        self.num_agents = self.num_teammates + 1

    #####################
    # PLASTIC Interface #
    #####################

    def likelihood_given_actions(self, state, teammates_actions):
        """
        Returns the likelihood of the model being the one which the agent is interacting with
        """
        policies = self.policies(state)
        probabilities = []
        for teammate_id, action in enumerate(teammates_actions):
            policy = policies[teammate_id]
            probability = policy[action]
            probabilities.append(probability)
        probabilities = np.array(probabilities)
        return np.multiply.reduce(probabilities)

    ###################
    # Abstract Methods #
    ###################

    @abstractmethod
    def policies(self, state):
        """
        Abstract Method
        Given a state, returns the teammates's policies
        """
        raise NotImplementedError()

    def simulate_teammates_actions(self, state):
        """
        Given a state, predicts a possible set of teammate actions,
        Given their policies
        """
        policies = self.policies(state)
        num_action = 4
        indices = [np.random.choice(range(num_action), p=pi) for pi in policies]
        return indices
