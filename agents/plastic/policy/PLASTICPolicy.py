from abc import ABC

import yaml

from agents.networks.DQN import DQN
from agents.networks.ReplayMemoryModel import ReplayMemoryModel
from agents.plastic.PLASTICPrior import PLASTICPrior


class PLASTICPolicy(PLASTICPrior, ABC):

    """ Encapsulates a team model and a dqn for that team model. """

    def __init__(self, teammates, num_teammates):
        self.dqn, self.teammates_model = self.setup_models()
        super().__init__(teammates, num_teammates)

    ####################
    # Abstract Methods #
    ####################

    def setup_models(self):
        raise NotImplementedError()

    @staticmethod
    def setup_dqn(trainable):
        with open("config.yaml", 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
            cfg = config["dqn"]
        dqn = DQN(trainable, config["learning rate"], cfg["discount factor"],
                  ReplayMemoryModel.parse_layer_blueprints(cfg["layers"]),
                  config["replay min batch"], config["replay memory size"])
        return dqn
