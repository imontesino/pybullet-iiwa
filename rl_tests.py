"""Functions to train using stable baselines 3 fo the kuka position control
environment"""

import itertools
import os
import shutil
import time

import imageio
import IPython
import numpy as np

# RL Algorithms and envs
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

# kuka environment for Position control
from envs.kukaEnvs import PCEnv


def delete_contents(folder):
    """Remove all contents and subdirectories in a given directory

    :param folder: the path of the directory to delete
    """
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


def visualize_agent(model):
    """Show the agent acting in a Visual environment

    :param model: a trained model of an agent
    """
    visualize_env = PCEnv(gui=True)
    obs = visualize_env.reset()
    for _ in range(2000):
        action = model.predict(obs, deterministic=True)
        obs, _, _, _ = visualize_env.step(action[0])
        time.sleep(1 / 240)

    visualize_env.close()


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env. Modified for KukaEnvs.PCEnv

    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = PCEnv(gui=False)
        env.seed(seed + rank)
        return env

    return _init


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
        [10, 1, 0.1],  # target smoothing coefficient
        ["auto", "auto_0.1", "auto_0.01"],  # entropy coefficient
    ]

    parm_test_set = tuple(itertools.product(*test_parameters))

    vec_env = SubprocVecEnv([make_env(i) for i in range(12)])

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
            # log_interval=1, # episodes (1 ep = 10000 ts)
            tb_log_name=name,
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

    env_gui = PCEnv(gui=True)
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


if __name__ == "__main__":

    sac_grid_search(timesteps=int(1e7))

