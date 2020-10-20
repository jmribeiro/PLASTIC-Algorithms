from abc import ABC, abstractmethod

import numpy as np
import yaml

from environment.PursuitState import PursuitState
from environment.utils import pursuit_datapoint
from yaaf.policies import deterministic_policy
from yaaf.agents import Agent


class PLASTICAgent(Agent, ABC):

    """ Plastic Agent - Has n possible team models m, identifies or learns the current one. """

    def __init__(self, name, num_teammates, world_size, learn_team=True, verbose=False):

        super().__init__(name)

        with open("config.yaml", 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        self.world_size = world_size
        self.num_teammates = num_teammates
        self.num_agents = self.num_teammates + 1

        self.learning_team = learn_team
        if self.learning_team:
            self.priors = [self.setup_learning_prior()]
            self.learning_prior = self.priors[0]
        else:
            self.priors = []

        self.belief_distribution = np.zeros((len(self.priors),)) + 1 / len(self.priors)

        assert np.sum(self.belief_distribution) == 1.0, "Something went wrong initializing the beliefs..."

        self.eta = config["eta"]

        self.verbose = verbose

    #####################
    # PLASTIC Interface #
    #####################

    def update_beliefs(self, pursuit_state, joint_actions):

        # Fetch teammate's actions
        teammates_actions = joint_actions[1:len(joint_actions) + 1]

        # P(a|m, s) for each m in beliefs
        P_a_m_s = [model.likelihood_given_actions(pursuit_state, teammates_actions) for model in self.priors]

        # Compute beliefs given new data
        for i in range(len(self.belief_distribution)):
            loss = 1 - P_a_m_s[i]
            self.belief_distribution[i] *= 1 - self.eta * loss
            self.belief_distribution[i] = float("{0:.5f}".format(self.belief_distribution[i]))

        # Normalize for distribution
        self.belief_distribution = self.belief_distribution / self.belief_distribution.sum()
        return self.belief_distribution

    def most_likely_model(self):
        """
        Given the belief distribution, picks a random model
        The higher a model belief, the higher the probability of it being picked
        """
        choice = self.belief_distribution.argmax()
        if self.verbose:
            print(f"Belief Dist.: {self.belief_distribution}")
            print("Most Likely Team: ", self.priors[choice].name)
        return self.priors[choice]

    #########
    # Agent #
    #########

    def policy(self, observation):
        pursuit_state = PursuitState.from_features(observation, self.world_size)
        action = self.select_action_according_to_model(pursuit_state, self.most_likely_model())
        return deterministic_policy(action, num_actions=4)

    def _reinforce(self, timestep):
        info = {}
        state, joint_actions, reward, next_state, terminal = pursuit_datapoint(timestep, self.world_size)
        beliefs = self.update_beliefs(state, joint_actions)
        info["belief distribution"] = {}
        for i, m in enumerate(self.priors):
            info["belief distribution"][m.name] = beliefs[i]
        info["learning prior"] = self.learning_prior.reinforce(state, joint_actions, reward, next_state, terminal)
        return info

    def save(self, directory):
        super().save(directory)
        np.save(f"{directory}/beliefs.npy", self.belief_distribution)
        self.learning_prior.save(directory)

    def save_learning_prior(self, directory, name, clear=True):
        if not self.learning_prior:
            print("WARN: Not learning any team", flush=True)
            return
        self.learning_prior.name = name
        self.learning_prior.save(directory)
        if clear:
            del self.learning_prior
            self.learning_prior = self.setup_learning_prior()
            self.priors[-1] = self.learning_prior

    def load_learnt_prior(self, directory, team_name):
        prior = self._load_prior_team(directory, team_name)
        self.priors = [prior] + self.priors
        del self.belief_distribution
        self.belief_distribution = np.zeros((len(self.priors),)) + 1 / len(self.priors)
        assert np.sum(self.belief_distribution) == 1.0

    def load(self, directory):
        self.learning_prior.load(directory)
        self.belief_distribution = np.load(f"{directory}/beliefs.npy")

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def select_action_according_to_model(self, pursuit_state, most_likely_model):
        """
        Returns the best learnt action for the state,
        assuming the teammates follow this prior model
        """
        raise NotImplementedError()

    @abstractmethod
    def setup_learning_prior(self):
        raise NotImplementedError()

    @abstractmethod
    def _load_prior_team(self, directory, name):
        raise NotImplementedError()
