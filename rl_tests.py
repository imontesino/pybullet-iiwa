"""Functions to train using stable baselines 3 fo the kuka position control
environment"""

import argparse
import itertools
import os
import time
from typing import List

import imageio
import numpy as np
import psutil

# RL Algorithms and envs
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from utils.folders import delete_contents

# kuka environment for Position control
from iiwa_fri_gym import TorqueSimEnv

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

        def traj(t): return np.array([x, y, z, 0, -np.pi, 0])
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


def baseline_sac(timesteps=int(1e6)):
    """Baseline result with default parameters for the SAC algorithm"""
    num_cpu = 18  # psutil.cpu_count()
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    eval_callback = EvalCallback(vec_env, best_model_save_path='./checkpoints/',
                                 log_path='./logs/', eval_freq=int(1e5),
                                 deterministic=True, render=False)

    checkpoint_callback = CheckpointCallback(save_freq=int(1e6), save_path='./checkpoints/',
                                             name_prefix='model_chkpt')

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    model = SAC(
        "MlpPolicy",
        vec_env,
        ent_coef="auto_0.1",
        verbose=2,
        tensorboard_log="test_log_dir"
    )

    model.learn(timesteps,
                tb_log_name="test_name_sac",
                callback=callback_list)

    model.save("models/sac_default_parameters_{}_timesteps".format(timesteps))


def further_training(model_file: str, timesteps: int = int(1e6), savefile: str=None):
    """Load an already train model and train it with random trajectories in its workspace"""
    if savefile is None:
        savefile = model_file.split("/")[-1]+"and_polars"

    num_cpu = psutil.cpu_count() - 1
    vec_env = SubprocVecEnv([make_env(
        i, rand_traj_training=True, reward_function="cartesian") for i in range(num_cpu)])

    eval_callback = EvalCallback(vec_env, best_model_save_path='./checkpoints/',
                                 log_path='./logs/', eval_freq=int(1e5),
                                 deterministic=True, render=False)

    checkpoint_callback = CheckpointCallback(save_freq=int(1e6), save_path='./checkpoints/',
                                             name_prefix='savefile')

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    model = SAC.load(model_file, env=vec_env)

    model.learn(timesteps,
                tb_log_name="savefile",
                callback=callback_list)

    model.save("models/"+savefile)
    print(model_file.split("."))
    print(savefile)


def random_trajectory_sac(timesteps=int(1e6), savefile: str=None):
    """Result with random trajectories to follow using SAC"""

    if savefile is None:
        savefile = f"sac_random_traj_{timesteps}_timesteps"

    num_cpu = psutil.cpu_count() - 1
    vec_env = SubprocVecEnv(
        [make_env(i, rand_traj_training=False) for i in range(num_cpu)])

    eval_callback = EvalCallback(vec_env, best_model_save_path='./checkpoints/',
                                 log_path='./logs/', eval_freq=int(1e5),
                                 deterministic=True, render=False)

    checkpoint_callback = CheckpointCallback(save_freq=int(1e6), save_path='./checkpoints/',
                                             name_prefix='model_chkpt')

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    model = SAC(
        "MlpPolicy",
        vec_env,
        ent_coef="auto_0.1",
        verbose=2,
        tensorboard_log="test_log_dir"
    )

    model.learn(timesteps,
                tb_log_name="sac_random_traj",
                callback=callback_list)

    model.save("models"+savefile)


def random_trajectory_sac_viz(timesteps=int(1e6)):
    """Visualize training with random trajectories to follow using SAC"""
    vec_env = TorqueSimEnv(gui=True, rand_traj_training=True)

    eval_callback = EvalCallback(vec_env, best_model_save_path='./checkpoints/',
                                 log_path='./logs/', eval_freq=int(1e5),
                                 deterministic=True, render=False)

    checkpoint_callback = CheckpointCallback(save_freq=int(1e6), save_path='./checkpoints/',
                                             name_prefix='model_chkpt')

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    model = SAC(
        "MlpPolicy",
        vec_env,
        ent_coef="auto_0.1",
        verbose=2,
        tensorboard_log="test_log_dir"
    )

    model.learn(timesteps,
                tb_log_name="sac_random_traj",
                callback=callback_list)

    model.save("models/sac_random_traj_{}_timesteps".format(timesteps))


def parse_args():
    parser = argparse.ArgumentParser(description='Test or watch the agents')

    parser.add_argument('--train', metavar='STEPS', type=int,
                        help='train for STEPS timesteps')

    parser.add_argument('--test', action='store_true',
                        help='Run the agent in an interactive environment')

    args = parser.parse_args()

    return args

def choose_model_file() -> str:
    """Chooose a model file from a list of files in the models save directory"""
    items = os.listdir("models")

    model_files: List[str] = []
    for name in items:
        if name.endswith(".zip"):
            model_files.append(name)

    print("----- Model Files ------")
    for i, file in enumerate(model_files):
       print(str(i)+": "+file)

    while True:
        user_option = int(input("Model to load: "))
        if user_option < len(model_files):
            break
        else:
            print("Wrong option")

    return "models/"+model_files[user_option].replace(".zip","")


if __name__ == "__main__":

    args = parse_args()

    if args.test is True:
        model_file = choose_model_file()
        model = SAC.load(model_file)
        interact_agent(model)

    elif args.train is not False:
        training_steps = args.train

        model_file = None
        while True:
            use_model = input("Use a previous model as starting point? [y/n]: ")
            if use_model.lower() == "y":
                model_file = choose_model_file()
                break
            elif use_model.lower() == "n":
                break

        savefile = input("Choose a name for the saved model file: ")

        if model_file is not None:
            further_training(model_file, timesteps=training_steps, savefile=savefile)
        else:
            random_trajectory_sac(timesteps=training_steps, savefile=savefile)

