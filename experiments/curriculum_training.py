"""Functions to train using stable baselines 3 fo the kuka position control
environment"""

import argparse

import psutil

# RL Algorithms and envs
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils.folders import choose_model_file
from utils.rl_helpers import interact_agent, make_env

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

    model.save("models/"+savefile)

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


def parse_args():
    parser = argparse.ArgumentParser(description='Test or watch the agents')

    parser.add_argument('--train', metavar='STEPS', type=int,
                        help='train for STEPS timesteps')

    parser.add_argument('--test', action='store_true',
                        help='Run the agent in an interactive environment')

    args = parser.parse_args()

    return args

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

