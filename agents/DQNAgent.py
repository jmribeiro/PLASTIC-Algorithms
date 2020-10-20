import yaml

from agents.networks.DQN import DQN
from agents.networks.ReplayMemoryModel import ReplayMemoryModel
from environment.PursuitState import PursuitState
from environment.utils import pursuit_datapoint
from yaaf.agents import Agent
from yaaf.policies import lazy_epsilon_greedy_policy, boltzmann_policy, random_policy, greedy_policy, linear_annealing


class DQNAgent(Agent):

    def __init__(self, world_size):
        super().__init__("DQN")
        with open("config.yaml", 'r') as stream: config = yaml.load(stream, Loader=yaml.FullLoader)
        self.dqn = DQN(True,
                       config["learning rate"],
                       config["dqn"]["discount factor"],
                       ReplayMemoryModel.parse_layer_blueprints(config["dqn"]["layers"]),
                       config["replay min batch"],
                       config["replay memory size"])

        self._policy = config["exploration policy"]
        self._initial_collect_steps = config["initial collect steps"]
        self._start_exploration_rate = config["start exploration rate"]
        self._end_exploration_rate = config["end exploration rate"]
        self.final_timesteps = config["final exploration timestep"]
        self.world_size = world_size

    #########
    # Agent #
    #########

    def policy(self, observation):

        if not self.trainable:
            return greedy_policy(self.q_values(observation))

        if self.total_training_timesteps <= self._initial_collect_steps:
            return random_policy(num_actions=4)
        elif self._policy == "epsilon-greedy":
            return lazy_epsilon_greedy_policy(lambda: self.q_values(observation), num_actions=4, epsilon=self.exploration_rate)
        elif self._policy == "boltzmann":
            return boltzmann_policy(self.q_values(observation), tau=self.exploration_rate)

    def q_values(self, observation):
        state = PursuitState.from_features(observation, self.world_size)
        features = state.features_relative_agent(agent_id=0)
        return self.dqn.predict(features)

    def _reinforce(self, timestep):
        exploration_parameter = "exploration rate" if self._policy == "epsilon-greedy" else "boltzmann temperature"
        return {
            "dqn": self.dqn.replay_fit(pursuit_datapoint(timestep, self.world_size)),
            exploration_parameter: self.exploration_rate
        }

    @property
    def exploration_rate(self):
        if not self.trainable: return 0.0
        elif self.total_training_timesteps <= self._initial_collect_steps: return 1.0
        else:
            return linear_annealing(
                self.total_training_timesteps - self._initial_collect_steps,
                self.final_timesteps,
                self._start_exploration_rate,
                self._end_exploration_rate
            )

    def save(self, directory):
        super().save(directory)
        self.dqn.save(directory)

    def load(self, directory):
        self.dqn.load(directory)
