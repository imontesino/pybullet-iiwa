import argparse
import itertools
import os

import imageio
import numpy as np
import psutil
# kuka environment for Position control
from iiwa_fri_gym import TorqueSimEnv
# RL Algorithms and envs
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils.folders import delete_contents
from utils.rl_helpers import make_env


def sac_grid_search(timesteps=int(1e6)):
    """Run a grid search for hyperparameter tunning on SAC

    :param timesteps: training timesteps for each point on hyperparameter grid
    """

    log_dir = "sac_tests_grid_search/{}_timesteps/".format(timesteps)
    os.makedirs(log_dir, exist_ok=True)
    if timesteps < 1e6:
        delete_contents(log_dir)

    # Create log dir
    model_dir = "models/sac_tests_grid_search/{}_timesteps/".format(timesteps)
    os.makedirs(model_dir, exist_ok=True)

    test_parameters = [
        [1, 0.1],  # target smoothing coefficient
        ["auto", "auto_0.1", "auto_0.01"],  # entropy coefficient
    ]

    parm_test_set = tuple(itertools.product(*test_parameters))

    num_cpu = psutil.cpu_count() - 1
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    for param in parm_test_set:
        tau = param[0]
        ent_coef = param[1]
        print("--- tau: {} Ent Coeff: {} ---".format(tau, ent_coef))
        name = "SAC_{}tau_{}ec".format(
            str(tau).replace(".", "0"), str(ent_coef).replace(".", "0")
        )
        vec_env.reset()
        model = SAC(
            "MlpPolicy",
            vec_env,
            tau=tau,
            verbose=1,
            ent_coef=ent_coef,
            tensorboard_log=log_dir,
        )

        model.learn(
            timesteps,
            tb_log_name=name
        )

        name = os.path.join(model_dir, name)
        model.save(name)


def sac_grid_watch(save=False, fps=30):
    """Watch the gridsearch results in robots along the XY plane

    :param save: save a gif of the visualization
    :param fps: fps of the saved gif
    """

    images = []
    timesteps = int(1e6)

    env_gui = TorqueSimEnv(gui=True)
    model_dir = "models/sac_tests_grid_search/{}_timesteps".format(timesteps)

    taus = [10, 1, 0.1]  # target smoothing coefficient
    ent_coefs = ["auto", "auto_0.1", "auto_0.01"]  # entropy coefficient

    for tau in taus:
        for ent_coef in ent_coefs:
            name = "SAC_{}tau_{}ec".format(
                str(tau).replace(".", "0"), str(ent_coef).replace(".", "0")
            )

            filename = os.path.join(model_dir, name)
            model = SAC.load(filename)
            env_gui.onscreen_title(name)
            obs = env_gui.reset()
            for i in range(8000):
                action = model.predict(obs, deterministic=True)
                obs, _, _, _ = env_gui.step(action[0])
                if i % int(240 / fps) == 0 and save:
                    img = env_gui.render(mode="rgb_array")
                    images.append(img)

    if save:
        imageio.mimsave(
            "algorithms_comparison.gif",
            [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
            fps=fps,
        )

    env_gui.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Test or watch the agents')

    parser.add_argument('--train', metavar='STEPS', type=int,
                        help=('perform a hyperparameter grid search for '
                              'STEPS timesteps in each point'))

    parser.add_argument('--test', action='store_true',
                        help='View a grid of agents trained on the diffrent parameters')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()

    if args.test is True:
        sac_grid_watch()

    elif args.search is not False:
        training_steps = args.train

        savefile = input("Choose a name for the saved model file: ")

        sac_grid_search(timesteps=training_steps)
