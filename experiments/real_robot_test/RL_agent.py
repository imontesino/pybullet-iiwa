import argparse
import os
import time

import numpy as np
from iiwa_fri_gym import TorqueSimEnv
from iiwa_fri_gym.fri_env import TorqueFriEnv
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def interact_agent_sim(agent: BaseAlgorithm):
    """Show the agent acting in a Visual environment

    :param agent: a trained agent of an agent
    """
    visualize_env = TorqueSimEnv(gui=True,
                                 episode_timesteps=2000,
                                 reward_function='cartesian')

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
        action = agent.predict(obs, deterministic=True)
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

def interact_agent_real(agent: BaseAlgorithm):
    """Show the agent acting in a Visual environment

    :param agent: a trained agent of an agent
    """
    visualize_env = TorqueSimEnv(gui=True,
                                 episode_timesteps=2000,
                                 reward_function='cartesian')

    p = visualize_env.p
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    x_id = p.addUserDebugParameter("target_x", -1, 1, 0.53)
    y_id = p.addUserDebugParameter("target_y", -1, 1, 0)
    z_id = p.addUserDebugParameter("target_z", -1, 1, 0.55)
    close_id = p.addUserDebugParameter("close", 1, 0, 1)
    close = False

    obs = visualize_env.reset()

    real_env = TorqueFriEnv(gui=True,
                            episode_timesteps=2000,
                            reward_function='cartesian')

    while not close:
        t_start = time.time()

        # Read User Parameters
        close = p.readUserDebugParameter(close_id) > 1
        x = p.readUserDebugParameter(x_id)
        y = p.readUserDebugParameter(y_id)
        z = p.readUserDebugParameter(z_id)

        # Send desired point to robot
        traj = lambda t : np.array([x, y, z, 0, -np.pi, 0])

        #visualize_env.trajectory = traj
        real_env.trajectory = traj

        action = agent.predict(obs, deterministic=True)
        obs, _, _, _ = real_env.step(action[0])
        #visualize_env.robot.set_joint_states(real_env.robot.get_joint_positions())

        t_end = time.time()

        time.sleep(max(0, 1/240-(t_end-t_start)))

    visualize_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Test or watch the agents')

    parser.add_argument('--real', action='store_true',
                        help='Run the code in the reak robot environment')

    parsed_args = parser.parse_args()

    return parsed_args

if __name__ == "__main__":

    args = parse_args()

    trained_agent = SAC.load("models/circle_agent")

    if args.real:
        interact_agent_real(trained_agent)
    else:
        interact_agent_sim(trained_agent)
