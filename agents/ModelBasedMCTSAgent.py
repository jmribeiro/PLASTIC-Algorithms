import yaml

from agents.networks.IndependentStochasticEnvironmentModel import IndependentStochasticEnvironmentModel
from agents.networks.TeamModel import TeamModel
from environment.PursuitState import PursuitState
from agents.networks.PerfectEnvironmentModel import PerfectEnvironmentModel
from agents.networks.ReplayMemoryModel import ReplayMemoryModel
from agents.plastic.model.HandcodedPLASTICModel import HandcodedPLASTICModel
from agents.search.PursuitMCTSNode import PursuitMCTSNode
from environment.utils import pursuit_datapoint

from yaaf.agents import Agent
from yaaf.policies import random_policy, lazy_epsilon_greedy_policy, boltzmann_policy, greedy_policy, linear_annealing


class ModelBasedMCTSAgent(Agent):

    def __init__(self, num_teammates, world_size, environment_model="perfect", teammates_model="greedy", name="Model Based MCTS", config=None):

        super().__init__(name)

        if config is None:
            with open("config.yaml", 'r') as stream:
                config = yaml.load(stream, Loader=yaml.FullLoader)

        self.world_size = world_size

        # ########## #
        # Parameters #
        # ########## #

        self._policy = config["exploration policy"]
        self._initial_collect_steps = config["initial collect steps"]
        self._start_exploration_rate = config["start exploration rate"]
        self._end_exploration_rate = config["end exploration rate"]
        self.final_timesteps = config["final exploration timestep"]

        self.mcts_iterations = config["mcts"]["iterations"]
        self.mcts_Cp = config["mcts"]["Cp"]
        self.mcts_max_rollout_depth = config["mcts"]["maximum rollout depth"]
        self.mcts_discount_factor = config["mcts"]["discount factor"]

        # ############ #
        # Setup Models #
        # ############ #

        if teammates_model == "greedy" or teammates_model == "teammate aware":
            self.teammates_model = HandcodedPLASTICModel(teammates_model, num_teammates, world_size)
        else:
            self.teammates_model = TeamModel(num_teammates=num_teammates, trainable=True)

        if environment_model == "perfect":
            self.environment_model = PerfectEnvironmentModel()
        elif environment_model == "learning" or environment_model == "stochastic":
            self.environment_model = IndependentStochasticEnvironmentModel(num_agents=num_teammates + 1, trainable=True,
                                                                           learning_rate=config["learning rate"],
                                                                           layers=ReplayMemoryModel.parse_layer_blueprints(config["environment model"]["layers"]),
                                                                           replay_batch_size=config["replay min batch"],
                                                                           replay_memory_size=config["replay memory size"])
        elif environment_model == "knows reward":
            self.environment_model = IndependentStochasticEnvironmentModel(num_agents=num_teammates + 1, trainable=True,
                                                                           learning_rate=config["learning rate"],
                                                                           layers=ReplayMemoryModel.parse_layer_blueprints(config["environment model"]["layers"]),
                                                                           replay_batch_size=config["replay min batch"],
                                                                           replay_memory_size=config["replay memory size"],
                                                                           learn_reward=False)
        else:
            assert isinstance(environment_model, ReplayMemoryModel)
            self.environment_model = environment_model

        self.train() if self.environment_model.trainable or self.teammates_model.trainable else self.eval()

    def policy(self, observation):

        if self.environment_model.name == "Perfect Environment Model" or not self.trainable:
            Q = self.mcts_q_values(observation)
            return greedy_policy(Q)

        if self.total_training_timesteps <= self._initial_collect_steps:
            return random_policy(num_actions=4)
        elif self._policy == "epsilon-greedy":
            return lazy_epsilon_greedy_policy(lambda: self.mcts_q_values(observation), 4, self.exploration_rate)
        elif self._policy == "boltzmann":
            mcts_q_values = self.mcts_q_values(observation)
            policy = boltzmann_policy(mcts_q_values, self.exploration_rate)
            return policy

    def mcts_q_values(self, observation):
        state = PursuitState.from_features(observation, self.world_size)
        root = PursuitMCTSNode(state, self.environment_model.simulate_transition, self.teammates_model.simulate_teammates_actions)
        root.uct_search(self.mcts_iterations, self.mcts_max_rollout_depth, self.mcts_Cp, self.mcts_discount_factor)
        return root.Q

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

    def _reinforce(self, timestep):
        info = {}
        datapoint = pursuit_datapoint(timestep, self.world_size)
        info["environment model"] = self.environment_model.replay_fit(datapoint)
        info["team model"] = self.teammates_model.replay_fit(datapoint)
        exploration_parameter = "exploration rate" if self._policy == "epsilon-greedy" else "boltzmann temperature"
        info[exploration_parameter] = self.exploration_rate
        return info
