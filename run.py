import argparse

from yaaf.visualization import LinePlot

from experiment_utils import main_run

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('agent', choices=["dummy", "greedy", "teammate_aware", "probabilistic_destinations", "dqn", "plastic_policy", "plastic_model"])
    parser.add_argument('-team', type=str, choices=["greedy", "mixed", "teammate_aware"], default="teammate_aware")
    parser.add_argument('-world_size', type=int, default=5)
    parser.add_argument('-n', type=int, default=1)
    parser.add_argument('-timesteps', type=int, default=5000)
    parser.add_argument('-eval', type=int, default=500)

    opt = parser.parse_args()

    opt.agent = opt.agent.replace("_", " ")
    opt.team = opt.team.replace("_", " ")
    world_size = opt.world_size, opt.world_size

    result = main_run(opt.agent, world_size, opt.team, opt.timesteps, opt.eval, 10, "default")

    plot = LinePlot(f"{opt.agent} on {opt.team} ({world_size})", x_label="Timestep", y_label=f"Prey Captures Every {opt.eval} timesteps", num_measurements=int(opt.timesteps/opt.eval), x_tick_step=opt.eval)
    plot.add_run(opt.agent, result)
    plot.show()

