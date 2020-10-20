import yaml

from agents.plastic.PLASTICAgent import PLASTICAgent
from agents.plastic.policy.LearningPLASTICPolicy import LearningPLASTICPolicy
from agents.plastic.policy.LearntPLASTICPolicy import LearntPLASTICPolicy
from yaaf.policies import action_from_policy, lazy_epsilon_greedy_policy, linear_annealing


class PLASTICPolicyAgent(PLASTICAgent):

    def __init__(self, num_teammates, world_size):
        super().__init__("Plastic Policy", num_teammates, world_size)
        with open("config.yaml", 'r') as stream: config = yaml.load(stream, Loader=yaml.FullLoader)
        self._start_exploration_rate = config["start exploration rate"]
        self._end_exploration_rate = config["end exploration rate"]
        self._final_timesteps = config["final exploration timestep"]

    #################
    # PLASTIC Agent #
    #################

    def select_action_according_to_model(self, state, most_likely_model):

        if isinstance(most_likely_model, LearningPLASTICPolicy):
            exploration_rate = linear_annealing(most_likely_model.total_timesteps, self._final_timesteps, self._start_exploration_rate, self._end_exploration_rate)
        else:
            exploration_rate = self._end_exploration_rate

        features_relative_agent = state.features_relative_agent(agent_id=0)
        q_function = lambda: most_likely_model.dqn.predict(features_relative_agent)

        policy = lazy_epsilon_greedy_policy(q_function, num_actions=4, epsilon=exploration_rate)
        action = action_from_policy(policy)

        return action

    def setup_learning_prior(self):
        return LearningPLASTICPolicy(self.num_teammates)

    def _load_prior_team(self, directory, name):
        return LearntPLASTICPolicy(directory, name, self.num_teammates)
