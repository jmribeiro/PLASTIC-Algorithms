import numpy as np
import yaml

from agents.plastic.model.LearningPLASTICModel import LearningPLASTICModel
from agents.plastic.model.LearntPLASTICModel import LearntPLASTICModel
from environment.Pursuit import Pursuit
from agents.plastic.PLASTICAgent import PLASTICAgent
from agents.plastic.model.HandcodedPLASTICModel import HandcodedPLASTICModel
from agents.search.PursuitMCTSNode import PursuitMCTSNode


class PLASTICModelAgent(PLASTICAgent):

    def __init__(self, num_teammates, world_size):
        super().__init__("Plastic model", num_teammates, world_size)
        with open("config.yaml", 'r') as stream: config = yaml.load(stream, Loader=yaml.FullLoader)["mcts"]

        self.world_size = world_size

        self.mcts_iterations = config["iterations"]
        self.mcts_Cp = config["Cp"]
        self.mcts_max_rollout_depth = config["maximum rollout depth"]
        self.mcts_discount_factor = config["discount factor"]

        self.load_handcoded_priors()

    def load_handcoded_priors(self):
        self.priors = [
            HandcodedPLASTICModel("greedy", self.num_teammates, self.world_size),
            HandcodedPLASTICModel("teammate aware", self.num_teammates, self.world_size)
        ] + self.priors
        self.belief_distribution = np.zeros((len(self.priors),)) + 1 / len(self.priors)
        assert np.sum(self.belief_distribution) == 1.0

    #################
    # PLASTIC Agent #
    #################

    def select_action_according_to_model(self, state, most_likely_model):
        root = PursuitMCTSNode(state, Pursuit.transition, most_likely_model.simulate_teammates_actions)
        action = root.uct_search(self.mcts_iterations, self.world_size[0], self.mcts_Cp, self.mcts_discount_factor)
        return action

    def setup_learning_prior(self):
        return LearningPLASTICModel(self.num_teammates)

    def _load_prior_team(self, directory, name):
        return LearntPLASTICModel(directory, name, self.num_teammates)
