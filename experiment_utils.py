import random
import shutil
import time
from os import listdir, path, remove
from os.path import isfile, join
from random import getrandbits, choice

import numpy as np

from agents.DQNAgent import DQNAgent
from agents.PLASTICModelAgent import PLASTICModelAgent
from agents.PLASTICPolicyAgent import PLASTICPolicyAgent

from agents.teammates.GreedyAgent import GreedyAgent
from agents.teammates.ProbabilisticDestinationsAgent import ProbabilisticDestinationsAgent
from agents.teammates.TeammateAwareAgent import TeammateAwareAgent
from environment.Pursuit import Pursuit
from metrics.PreyCapturesEveryTimesteps import PreyCapturesEveryTimesteps

from yaaf.agents import RandomAgent
from yaaf.execution import TimestepRunner
from yaaf import mkdir

RESULT_DIR = "resources/results"


def find_agent_runs(experiment_dir, agent):
    directory = f"{experiment_dir}/{agent}"
    mkdir(directory)
    runs_done = [file for file in listdir(directory) if isfile(join(directory, file)) and ".running" not in file]
    runs_running = [file for file in listdir(directory) if isfile(join(directory, file)) and ".running" in file]
    return len(runs_done) + len(runs_running)


def find_available_tasks(agents, world_size, team, runs_per_agent, experiment_name, verbose=True):
    tasks = {}
    directory = f"{RESULT_DIR}/{experiment_name}/{world_size[0]}x{world_size[1]}/{team}"
    for agent in agents:
        runs_so_far = find_agent_runs(directory, agent)
        runs_needed = max(runs_per_agent - runs_so_far, 0)
        if verbose:
            print(f"{agent} w/ {team} in {world_size[0]}x{world_size[1]}, runs: {runs_so_far}, runs needed: {runs_needed}", flush=True)
        if runs_needed > 0:
            tasks[agent] = runs_needed
    return tasks

# ########## #
# START MAIN #
# ########## #


def do_task(agents, world_size, team, runs_per_agent, timesteps, eval_interval, log_interval, experiment_name):
    time.sleep(random.randint(0, 2))
    available_tasks = find_available_tasks(agents, world_size, team, runs_per_agent, experiment_name, verbose=False)
    agents = list(available_tasks.keys())
    if len(agents) == 0: return
    agent = choice(agents)
    main_run(agent, world_size, team, timesteps, eval_interval, log_interval, experiment_name)


def main_run(agent_name, world_size, team, timesteps, eval_interval, log_interval, experiment_name):

    # Run preparations
    directory = f"{RESULT_DIR}/{experiment_name}/{world_size[0]}x{world_size[1]}/{team}/{agent_name}"
    mkdir(directory)

    # Temporary run indicator
    run_id = getrandbits(128)
    tmp = f"{directory}/{run_id}.running"
    np.save(tmp, np.zeros(2))

    try:

        print(f"***Starting fresh agent {agent_name}***", flush=True)
        agent = setup_agent(agent_name, world_size)

        print(f"***Pretraining adhoc agent '{agent_name}'***", flush=True)
        if agent_name == "adhoc" or agent_name == "plastic policy":
            teams_to_pretrain = ["greedy", "teammate aware"] if team == "mixed" else [team]
            pretrain_adhoc_agent(agent, world_size, timesteps, eval_interval, log_interval, teams_to_pretrain, experiment_name)

        print(f"***Running***", flush=True)
        metric = PreyCapturesEveryTimesteps(eval_interval, verbose=True, log_interval=log_interval)
        env = Pursuit(teammates=team, world_size=world_size)
        runner = TimestepRunner(timesteps, agent, env, observers=[metric])
        runner.run()

        print(f"***Done: {metric.result()}***", flush=True)
        main_result = metric.result()
        run_filename = f"{directory}/{run_id}"
        np.save(run_filename, main_result)

        if path.exists(tmp + ".npy"):
            remove(tmp + ".npy")

        return main_result

    except KeyboardInterrupt:
        pass




def pretrain_adhoc_agent(agent, world_size, timesteps, eval_interval, log_interval, teams_to_pre_train, experiment_name):

    tmp_dir = f"tmp_{getrandbits(64)}_{agent.name}"
    shutil.rmtree(tmp_dir, ignore_errors=True)

    for team in teams_to_pre_train:
        dir = f"{RESULT_DIR}/pretrains_{experiment_name}/{world_size[0]}x{world_size[0]}/{team.lower()}/{agent.name.lower()}"
        print(f"***{agent.name}'s prior population: {team} team***", flush=True)
        metric = PreyCapturesEveryTimesteps(eval_interval, verbose=True, log_interval=log_interval)
        runner = TimestepRunner(timesteps, agent, Pursuit(team, 3, world_size), observers=[metric])
        runner.run()
        print(f"***{agent.name}'s prior population: {team} team: Done -> {metric.result()}***", flush=True)
        agent.save_learning_prior(tmp_dir, team)
        mkdir(dir)
        np.save(f"{dir}/{getrandbits(64)}.npy", metric.result())

    for team in teams_to_pre_train:
        agent.load_learnt_prior(f"{tmp_dir}/{team}", team)

    shutil.rmtree(tmp_dir, ignore_errors=True)

# ######## #
# END MAIN #
# ######## #


def setup_agent(agent_name, world_size):

    agents = {

        # Model Free
        "dqn": lambda: DQNAgent(world_size),

        # AdHoc
        "plastic model": lambda: PLASTICModelAgent(3, world_size),
        "plastic policy": lambda: PLASTICPolicyAgent(3, world_size),

        # Handcoded
        "teammate aware": lambda: TeammateAwareAgent(0, world_size),
        "greedy": lambda: GreedyAgent(0, world_size),
        "dummy": lambda: RandomAgent(4),
        "probabilistic destinations": lambda: ProbabilisticDestinationsAgent(0, world_size),

    }

    return agents[agent_name]()
