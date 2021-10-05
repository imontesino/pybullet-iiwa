import math
import os
import time

import numpy as np
from stable_baselines3 import SAC

from iiwa_fri_gym.sim_env import PbEnvServer

from experiments.utils.rl_helpers import steps2str


timesteps = 1000000
path = f"models/{timesteps}_timesteps"

taus = [1, 0.1, 10] # target smoothing coefficient
ent_coefs = ["auto_0.1", "auto", "auto_0.01"]  # entropy coefficient

models = np.empty((len(taus),len(ent_coefs)),dtype=object)
for i, tau in enumerate(taus):
    for j, ent_coef in enumerate(ent_coefs):
        # filename from hyperparameters
        filename = "SAC_{}tau_{}ec".format(str(tau).replace(".", "0"),
                                             ent_coef.replace(".", "0"))

        model_file = os.path.join(path, filename)

        name = "tau: {} ent: {}".format(tau, ent_coef)
        buffer = {"name": name, "file": model_file+".zip"}
        models[i][j] = buffer

envs = []
n_envs = models.size
n_envs_x = models.shape[0]
n_envs_y = models.shape[1]

L = 1.5  # separation between robots
ofst_x = (L/2)*((n_envs_x+1) % 2)
ofst_y = (L/2)*((n_envs_y+1) % 2)

starting_points = np.zeros((models.shape[0],models.shape[1],3))
for i in range(models.shape[0]):
    for j in range(models.shape[1]):
        starting_points[i][j][0] = ofst_x+L*math.ceil(0.5*j)*(-1)**j
        starting_points[i][j][1] = ofst_y+L*math.ceil(0.5*i)*(-1)**i
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
agents = []
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

    while time.time()-t_start < 1/240 :  # slow down to real time
        time.sleep(1/240)
    server.step()

for env in envs:
    env.reset()
