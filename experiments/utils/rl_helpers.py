"""Functions for rl agent visualization, interaction, creation, etc."""

import time

import numpy as np
from iiwa_fri_gym.sim_env import TorqueSimEnv


#TODO make functions env independent

def visualize_agent(model):
    """Show the agent acting in a Visual environment

    :param model: a trained model of an agent
    """
    visualize_env = TorqueSimEnv(gui=True, rand_traj_training=True,
                          episode_timesteps=2000, reward_function='cartesian')
    obs = visualize_env.reset()
    for _ in range(2000):
        action = model.predict(obs, deterministic=True)
        obs, _, _, _ = visualize_env.step(action[0])
        time.sleep(1 / 240)

    visualize_env.close()


def interact_agent(model):
    """Show the agent acting in a Visual environment

    :param model: a trained model of an agent
    """
    visualize_env = TorqueSimEnv(gui=True, rand_traj_training=True,
                          episode_timesteps=2000, reward_function='cartesian')
    p = visualize_env.p
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    x_id = p.addUserDebugParameter("target_x", -1, 1, 0.53)
    y_id = p.addUserDebugParameter("target_y", -1, 1, 0)
    z_id = p.addUserDebugParameter("target_z", -1, 1, 0.55)
    close_id = p.addUserDebugParameter("close", 1, 0, 1)
    close = False

    obs = visualize_env.reset()

    while not close:
        t_start = time.time()

        # Read User Parameters
        close = p.readUserDebugParameter(close_id) > 1
        action = model.predict(obs, deterministic=True)
        x = p.readUserDebugParameter(x_id)
        y = p.readUserDebugParameter(y_id)
        z = p.readUserDebugParameter(z_id)

        traj = lambda t : np.array([x, y, z, 0, -np.pi, 0])
        visualize_env.trajectory = traj
        visualize_env.robot.torque_control(action[0])
        obs, _, _, _ = visualize_env.step(action[0])
        t_end = time.time()

        time.sleep(max(0, 1/240-(t_end-t_start)))

    visualize_env.close()


def make_env(rank, seed=0, *args, **kwargs):
    """
    Utility function for multiprocessed env. Modified for KukaEnvs.TorqueSimEnv

    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :param args: positional arguments passed to the env
    :param kwargs: keyword arguments passed to the env
    """

    def _init():
        env = TorqueSimEnv(*args, **kwargs)
        env.seed(seed + rank)
        return env

    return _init


def steps2str(steps, timestep = 1./240.) -> str:
    """Transform a trained timesteps to a time trained string

    Example:
        steps2str(1.6e7) -> '13 hours, 53 minutes'
    """
    seconds = (steps*(timestep)) % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    if hours > 0:
        time_str = f"{hours:.0f} hours, {minutes:.0f} minutes"
    elif minutes > 0:
        time_str = f"{minutes:.0f} minutes, {seconds:.0f} seconds"
    else:
        time_str = f"{seconds:.2f} seconds"
    return time_str
