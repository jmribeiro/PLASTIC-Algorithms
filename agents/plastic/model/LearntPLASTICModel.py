from agents.networks.TeamModel import TeamModel
from agents.plastic.PLASTICPrior import PLASTICPrior


class LearntPLASTICModel(PLASTICPrior):

    """ Loads a neural-learnt model m """

    def __init__(self, directory, teammates, num_teammates):
        self.teammates_model = TeamModel(num_teammates=num_teammates, trainable=False)
        self.teammates_model.load(directory, load_memory=False)
        super().__init__(teammates, num_teammates)

    #################
    # PLASTIC Prior #
    #################

    def policies(self, state):
        return self.teammates_model.policies(state)