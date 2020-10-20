from agents.networks.TeamModel import TeamModel
from agents.plastic.PLASTICPrior import PLASTICPrior
from yaaf import mkdir


class LearningPLASTICModel(PLASTICPrior):

    def __init__(self, num_teammates):
        self.teammates_model = TeamModel(num_teammates, trainable=True)
        super().__init__("new", num_teammates)

    ############################
    # Learning Prior Interface #
    ############################

    def reinforce(self, state, joint_actions, reward, next_state, terminal):
        info = {}
        datapoint = state, joint_actions, reward, next_state, terminal
        info["team model"] = self.teammates_model.replay_fit(datapoint)
        return info

    def save(self, directory):
        prior_dir = f"{directory}/{self.name}"
        mkdir(prior_dir)
        self.teammates_model.save(prior_dir)

    def load(self, directory):
        prior_dir = f"{directory}/{self.name}"
        self.teammates_model.load(prior_dir)

    #################
    # PLASTIC Prior #
    #################

    def policies(self, state):
        return self.teammates_model.policies(state)
