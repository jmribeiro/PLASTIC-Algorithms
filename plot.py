import argparse
import random
import numpy as np
from matplotlib.markers import CARETLEFT, CARETRIGHT, CARETUP

from yaaf import subdirectories, files, mkdir, rmdir
from yaaf.visualization import LinePlot

RESULT_DIR = "resources/results"
PLOT_DIR = "resources/plots"

def alias(agent):
    agent_name = "PLASTIC Model" if agent == "plastic model" else agent
    agent_name = "PLASTIC Policy" if agent_name == "plastic policy" else agent_name
    agent_name = "Random Policy" if agent_name == "dummy" else agent_name
    agent_name = "Greedy Policy" if agent_name == "greedy" else agent_name
    agent_name = "Teammate Aware Policy" if agent_name == "teammate aware" else agent_name
    agent_name = "Probabilistic Destinations Policy" if agent_name == "probabilistic destinations" else agent_name
    agent_name = "Model-Free Algorithm (DQN)" if agent_name == "dqn" else agent_name
    return agent_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('-eval', type=int, default=500)
    parser.add_argument('-confidence', type=float, default=0.99)
    parser.add_argument('-show', action="store_true")

    opt = parser.parse_args()

    colors = {
        "PLASTIC Policy": "r",
        "PLASTIC Model": "b",

        "Greedy Policy": "crimson",
        "Teammate Aware Policy": "salmon",
        "Probabilistic Destinations Policy": "darkorchid",

        "Random Policy": "darkslategray",
        "Model-Free Algorithm (DQN)": "salmon",
    }

    markers = {
        "PLASTIC Policy": "^",
        "PLASTIC Model": "v",

        "Greedy Policy": CARETLEFT,
        "Teammate Aware Policy": CARETRIGHT,
        "Probabilistic Destinations Policy": CARETUP,

        "Random Policy": "o",
        "Model-Free Algorithm (DQN)": "o",

    }

    ignored_keywords = []
    excluded_agents = []
    experiments = subdirectories(RESULT_DIR)

    rmdir(PLOT_DIR)
    mkdir(PLOT_DIR)

    for experiment in experiments:

        if True in [keyword in experiment for keyword in ignored_keywords]:
            continue

        for world_size in ["5x5", "10x10", "20x20"]:

            timesteps = int(world_size.split("x")[0]) * 1000

            if world_size != "5x5": ymin, ymax = 0, 50

            for team in ["greedy", "teammate aware", "mixed"]:

                if "inverted" in experiment or "mixed" in team:
                    ymin, ymax = 0, 70
                if world_size == "10x10":
                    ymin, ymax = 0, 42
                elif world_size == "20x20":
                    ymin, ymax = 0, 25
                else:
                    ymin, ymax = 0, 110

                plot = None
                team_dir = f"{RESULT_DIR}/{experiment}/{world_size}/{team}"
                agents = subdirectories(team_dir)
                for agent in agents:
                    if agent in excluded_agents:
                        continue
                    agent_dir = f"{team_dir}/{agent}"
                    runs = [file for file in files(agent_dir) if ".running" not in file and ".npy" in file]
                    if len(runs) == 0:
                        continue
                    sample = min(len(runs), opt.n)
                    runs = random.sample(runs, sample)
                    run = np.load(f"{agent_dir}/{runs[0]}")
                    if plot is None:
                        name = f"Experiment: {experiment}\n{team} ({world_size})"
                        plot = LinePlot(f"{name}", "Timesteps",
                                    "Prey captures every 500 timesteps", len(run), opt.eval, confidence=opt.confidence,
                                    ymin=ymin, ymax=ymax, colors=colors)
                    agent_name = alias(agent)
                    [plot.add_run(agent_name, np.load(f"{agent_dir}/{run}"), marker=markers[agent_name]) for run in runs]
                if plot is not None and plot.has_runs:
                    if opt.show:
                        plot.show()
                    else:
                        plot_dir = f"{PLOT_DIR}/{world_size}"
                        mkdir(f"{plot_dir}")
                        name = f"{experiment}_{team.replace(' ', '_')}"
                        plot.savefig(f"{plot_dir}/{name}.pdf")
                        mkdir(f"{plot_dir}/png")
                        plot.savefig(f"{plot_dir}/png/{name}.png")
