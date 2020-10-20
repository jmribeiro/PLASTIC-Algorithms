from agents.networks.TeamModel import TeamModel
from agents.plastic.policy.PLASTICPolicy import PLASTICPolicy


class LearntPLASTICPolicy(PLASTICPolicy):

    """ Loads a neural-learnt model m and its respective dqn """

    def __init__(self, directory, teammates, num_teammates):
        self.directory = directory
        self.num_teammates = num_teammates
        self.name = teammates
        super().__init__(teammates, num_teammates)

    ##################
    # PLASTIC Policy #
    ##################

    def setup_models(self):
        dqn = self.setup_dqn(trainable=False)
        teammates_model = TeamModel(num_teammates=self.num_teammates, trainable=False)
        dqn.load(self.directory, load_memory=False)
        teammates_model.load(self.directory, load_memory=False)
        return dqn, teammates_model

    ##################
    # PLASTIC Policy #
    ##################

    def policies(self, state):
        return self.teammates_model.policies(state)
