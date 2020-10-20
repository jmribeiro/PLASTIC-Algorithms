from agents.plastic.PLASTICPrior import PLASTICPrior
from agents.teammates.GreedyAgent import GreedyAgent
from agents.teammates.TeammateAwareAgent import TeammateAwareAgent


class HandcodedPLASTICModel(PLASTICPrior):

    def __init__(self, teammates, num_teammates, world_size):
        super().__init__(teammates, num_teammates)
        self.total_teammates = num_teammates
        self.world_size = world_size
        self.trainable = False
        self.teammates = [self.spawn_handcoded_teammate(t + 1) for t in range(num_teammates)]

    def spawn_handcoded_teammate(self, id):
        if self.name == "greedy": return GreedyAgent(id, self.world_size)
        elif self.name == "teammate aware": return TeammateAwareAgent(id, self.world_size)
        else: raise ValueError(f"Invalid handcoded team for PLASTIC Model - Available teams are greedy and teammate aware")

    #################
    # PLASTIC Prior #
    #################

    def policies(self, state):
        return [agent.policy(state.features()) for agent in self.teammates]

    def replay_fit(self, datapoint):
        pass
