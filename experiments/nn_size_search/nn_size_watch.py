"""Watch the results of the nn size search agents arraged in a grid"""

# RL Algorithms and envs
import os
import time
from typing import List

import numpy as np
from iiwa_fri_gym.sim_env import PbEnvServer, TorqueSimEnv
from stable_baselines3 import SAC

from utils.rl_helpers import steps2str

def nn_size_watch(timesteps=int(1e7)):
    """Try diffrent net architectures

    :param timesteps: training timesteps for each point on hyperparameter grid
    """

    path = f"models/{timesteps}_timesteps"

    # Policy network layers
    pi_layers = [
        [64, 64],
        [343, 343],
        [64, 64, 64],
        [64, 343, 64],
    ]

    # Q-function layers
    qf_layers = [
        [64, 64],
        [343, 343],
        [64, 64, 64],
        [64, 343, 64],
    ]

    models = np.empty((len(pi_layers),len(qf_layers)),dtype=object)

    for i, pi in enumerate(pi_layers):
        for j, qf in enumerate(qf_layers):
            # filename from hyperparameters
            filename = f"SAC_pi_{len(pi)}.{pi[1]}__qf_{len(qf)}.{qf[1]}"

            model_file = os.path.join(path, filename)

            name = f"pi: {pi} \n qf: {qf}"
            buffer = {"name": name, "file": model_file+".zip"}
            models[i][j] = buffer

    envs: List[TorqueSimEnv] = []
    n_envs = models.size
    n_envs_x = models.shape[0]
    n_envs_y = models.shape[1]

    L = 1.5  # separation between robots
    ofst_x = (L/2)*((n_envs_x+1) % 2)
    ofst_y = (L/2)*((n_envs_y+1) % 2)

    starting_points = np.zeros((models.shape[0],models.shape[1],3))
    for i in range(models.shape[0]):
        for j in range(models.shape[1]):
            starting_points[i][j][0] = ofst_x+L*np.ceil(0.5*j)*(-1)**j
            starting_points[i][j][1] = ofst_y+L*np.ceil(0.5*i)*(-1)**i
            starting_points[i][j][2] = 0

    # start the pybullet server
    server = PbEnvServer()
    title = f"Trained: tau vs ent_coef ({steps2str(timesteps)})"
    server.onscreen_title(title)

    for row in starting_points:
        for start_position in row:
            env = server.spawn_env(start_position=start_position) # 1 env == 1 robot
            env.reset()
            envs.append(env)


    # Get models for algorithms saved
    agents: List[SAC] = []
    obs = [None for _ in range(n_envs)]

    for i, model in enumerate(models.flatten()):
        agent = SAC.load(model["file"], policy="MlpPolicy")
        agents.append(agent)
        envs[i].onscreen_title(model["name"], text_size=0.2)

        # initial observations for each agent
        obs[i] = envs[i].reset()

    for _ in range(2000):
        t_start = time.time()

        for i, agent in enumerate(agents):
            env = envs[i]
            action = agent.predict(obs[i], deterministic=True)
            obs[i], _, _, _ = env.step(action[0])

        while(time.time()-t_start < 1/240):  # slow down to real time
            time.sleep(1/24000)
        server.step()

    for env in envs:
        env.reset()

if __name__ == "__main__":
    nn_size_watch()
