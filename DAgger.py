import argparse

import time
import os
import pickle
import numpy as np
from tensorboard_logging import Logger

# kuka environment for Position control
from envs.kukaEnvs import PCEnv

# Save GIF of agent
import imageio

# Debugging
import IPython
import timing
import time

# Prevent deprecation warnings for Tensorflow 1
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import os, shutil

def import_sb():
    "Avoid long imports before parsing"
    # RL Algorithms and envs
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, \
                                             LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C

# Pre-training (Behavior Cloning)
from stable_baselines.gail import ExpertDataset


def delete_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

timer = timing.timer()

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

# Custom MLP policy of three layers of size 128 each
class CustomMLPPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMLPPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[24, 150, 150,
                                                     dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")

def test_agent(model):
    "Run the agent and recover its trajectory"
    traj = []
    obs = model.env.reset()
    traj.append(obs[0])
    action = model.predict(obs, deterministic=False)
    for _ in range(5000):
        action = model.predict(obs, deterministic=True)
        obs, _, _, _ = model.env.step(action[0]) # PPO2 vectorizes the env
        traj.append(obs[0]) # observation is vectorized
    model.env.reset()
    return traj

def visualize_agent(model):
    visualize_env = PCEnv(render=True)
    obs = visualize_env.reset()
    action = model.predict(obs)
    for _ in range(5000):
        action = model.predict(obs)
        obs, _, _, _ = visualize_env.step(action[0])  # PPO2 vectorizes the env
        time.sleep(1/240)
    visualize_env.close()

def save_gif(model, filename="DAgger_agent.gif", time=5, fps=30):
    images = []
    obs = model.env.reset()
    img = model.env.render(mode='rgb_array')
    images.append(img)
    for i in range(int(time*240)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _ ,_ = model.env.step(action)
        # fps is lower than render timestep, skip frames for real-time gif
        if i % int(240/fps) == 0:
            img = model.env.render(mode='rgb_array')
            images.append(img)

    # Save to Gif File
    imageio.mimsave(
        filename,
        [np.array(img) for i, img in enumerate(images) if i%2 == 0],
        fps=fps)

def Dagger(env, algorithm, policy, iterations,
           batch_size=256,
           episodes_per_it=1,
           epochs_per_dataset=10,
           gif_interval=None):

    log_dir = "log/DAgger"
    os.makedirs(log_dir, exist_ok=True)
    tb_log = Logger(log_dir)

    # Save a gif of the model every 10% of the total number of iterations
    if gif_interval is None:
        # Prevent modulo by zero
        if iterations < 10:
            gif_interval = 1
        else:
            gif_interval = int(iterations / 20)

    # Create log dir
    gif_dir = "gifs/DAgger"
    os.makedirs(gif_dir, exist_ok=True)

    # Create an agent in the environment
    model = algorithm(policy,
                      env,
                      verbose=1,
                      tensorboard_log="log/"
                      )

    # Generate the first expert Dataset
    dataset = env.generate_expert_traj(num_episodes=1, episode_timesteps=5000)

    dataset_info = {
        "num_traj":[],
        "num_transition":[],
        "avg_ret":[],
        "std_ret":[]
    }

    print()
    for i in range(iterations):
        print("--- Iteration {} Performance ---".format(i))
        t_it_start = time.time()

        # Load the expert trajectories created by PID
        exp_dataset = ExpertDataset(
            traj_data=dataset,
            randomize=True,
            batch_size=batch_size,
            verbose=0)

        # Behavior Cloning on the dataset
        t_pretrain_start = time.time()
        model.pretrain(exp_dataset, n_epochs=epochs_per_dataset)
        t_pretrain_end = time.time()
        print("Pretraining Time {:.0f} s".format(t_pretrain_end-t_pretrain_start))


        # Dict for storing episode data as generated by expert
        buffer = {
            "actions": np.array([]),
            "episode_returns": np.array([]),
            "rewards": np.array([]),
            "obs": np.array([]),
            "episode_starts": np.array([])
        }

        # TODO Parallelize with VecEnv
        t_test_start = time.time()
        for n in range(episodes_per_it):
            # Run agent and record its trajectory
            traj = test_agent(model)

            # Get expert dataset from that trajectory
            new_trajectory = env.generate_expert_from_traj(traj)

            # Add that episode to the a buffer
            if n == 0:
                for key in buffer.keys():
                    buffer[key] = new_trajectory[key]
            else:
                for key in buffer.keys():
                    buffer[key] = np.concatenate((buffer[key],
                                                  new_trajectory[key]))
        t_test_end = time.time()

        # Add to the pretraining Dataset
        for key in dataset.keys():
            dataset[key] = np.concatenate((dataset[key],buffer[key]))

        # Save iteration performance data
        returns = buffer["episode_returns"]
        avg_ret = sum(returns) / len(returns)
        std_ret = np.std(np.array(returns))
        dataset_info["avg_ret"].append(avg_ret)
        dataset_info["std_ret"].append(std_ret)

        print("Episode Returns -> avg_ret: {:.0f} std_ret {:.0f}".format(avg_ret,std_ret))
        print("Testing Time {:.0f} s".format(t_test_end-t_test_start))
        tb_log.log_scalar('avg_ret', avg_ret, i)

        if gif_interval != 0 and (i % gif_interval) == 0:
            t_gif_start = time.time()
            save_gif(model, time=5, filename="gifs/DAgger/DAgger_iteration_{}.gif".format(i))
            t_gif_end = time.time()
            print("Gif saving {:.0f} s".format(t_gif_end-t_gif_start))

        t_it_end = time.time()
        print("Total Time {:.0f} s".format(t_it_end-t_it_start))
        print()

    with open('training_data/DAgger/dagger_training_data_{}_iterations.pkl'.format(iterations),'wb') as f:
        pickle.dump(dataset_info, f)

    np.savez("PID_expert_path.npz", **dataset)
    return model

def make_env(env_type):
    def _init():
        return env_type
    return _init

def parse_args():
    my_parser = argparse.ArgumentParser(
        description='Imitation learning of position control using th DAgger Algorithm')

    my_parser.add_argument('--test',
                        help='Test the last trained model',
                        action="store_true")

    my_parser.add_argument('--iterations', '-i',
                        type=int,
                        help='Iterartions of the DAgger algorithm',
                        default=15)

    my_parser.add_argument('--rl_timesteps', '-r',
                        type=int,
                        help='Run a DeepRL algorithm after dagger for RL_TIMESTEPS timesteps',
                        default=0)

    my_parser.add_argument('--gif_interval', '-g',
                        type=int,
                        help='Interval at wich to save a gif of the trained agent',
                        default=0)

    my_parser.add_argument('--gui',
                        help='Show a window view of the robot during the program (slows down traning)',
                        action="store_true")

    # Execute parse_args()
    args = my_parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    # DAgger parameters
    test = args.test
    dagger_iterations = args.iterations
    gif_interval = args.gif_interval

    # DeepRL after DAgger
    algorithm = A2C
    policy = CustomMLPPolicy
    rl_timesteps = args.rl_timesteps
    training_time = int(1e5)
    tensorboard_log="./{}_{}/".format(algorithm.__name__, policy.__name__)

    # Create the environment
    env = PCEnv(gui=args.gui)

    if not test:
        # Clean up last run results
        delete_contents("gifs/")
        delete_contents("log/")
        delete_contents("models/")

        # Learn with Imitation Learning using DAgger
        model = Dagger(env, algorithm, policy, dagger_iterations,
                       epochs_per_dataset=100,
                       gif_interval=gif_interval,
                       batch_size=8*4096)

        model_dir = "models/DAgger"
        os.makedirs(model_dir, exist_ok=True)
        model_name="Dagger_{}_iterations".format(dagger_iterations)
        model.save(os.path.join(model_dir+model_name))
        save_gif(model, filename="gifs/{}.gif".format(model_name))

        # Optimize further with RL
        if rl_timesteps > 0:
            model.learn(rl_timesteps)
            model_name="DAgger_and_{}_{:.0f}e5_timesteps".format(algorithm.__name__ ,
                                                            rl_timesteps/10000)
            model.save("models/"+model_name)
            save_gif(model, filename="gifs/{}.gif".format(model_name))

        print("Training Done!")

    else:
        env = DummyVecEnv([make_env(env)])
        model_name = "Dagger_model"
        model = algorithm.load("models/"+model_name, env=env)
        t_gif_start = time.time()
        save_gif(model, filename="gifsave_test.gif")
        t_gif_end = time.time()
        print("Gif creating time: {:.0f}s".format(t_gif_end-t_gif_start))

    env.close()
