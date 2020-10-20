from agents.plastic.PLASTICAgent import PLASTICAgent
from agents.plastic.model.LearningPLASTICModel import LearningPLASTICModel
from agents.plastic.model.LearntPLASTICModel import LearntPLASTICModel
from agents.teammates.GreedyAgent import GreedyAgent
from agents.teammates.TeammateAwareAgent import TeammateAwareAgent

import numpy as np

from yaaf.evaluation import Metric
import scipy.stats


class PLASTICTeammate(PLASTICAgent):

    def __init__(self, type, num_teammates, world_size):

        super(PLASTICTeammate, self).__init__("Plastic teammate", num_teammates, world_size)

        if type == "greedy":
            self._underlying_agent = GreedyAgent(0, world_size)
        elif type == "teammate aware" or type == "mixed":
            self._underlying_agent = TeammateAwareAgent(0, world_size)
        else:
            raise ValueError()

    def select_action_according_to_model(self, pursuit_state, most_likely_model):
        return self._underlying_agent.action(pursuit_state.features())

    def setup_learning_prior(self):
        return LearningPLASTICModel(self.num_teammates)

    def _load_prior_team(self, directory, name):
        return LearntPLASTICModel(directory, name, self.num_teammates)


class PLASTICAnalyzer(Metric):

    def __init__(self):

        super(PLASTICAnalyzer, self).__init__("PLASTIC Analyzer")

        self._entropy = []
        self._beliefs = []
        self._team_names = None

    def reset(self):
        self._entropy = []

    def __call__(self, timestep):
        info = timestep.info
        for key in info:
            if "Plastic" in key or key == "Adhoc":
                agent_info = info[key]
                belief_distribution = agent_info["belief distribution"]
                if self._team_names is None:
                    self._team_names = list(belief_distribution.keys())
                beliefs = np.array([belief_distribution[team] for team in self._team_names])
                entropy = scipy.stats.entropy(beliefs)
                self._beliefs.append(beliefs)
                self._entropy.append(entropy)

        return self._entropy[-1]

    def result(self):
        return np.array(self._entropy)

    def team_names(self):
        return self._team_names

    def beliefs(self):
        return np.array(self._beliefs)
