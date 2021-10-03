import math
import os
import time

import numpy as np
from stable_baselines3 import SAC

from iiwa_fri_gym.sim_env import PbEnvServer

def steps2str(steps):
    seconds = (steps*(1/240)) % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    if hours > 0:
        time_str = "{:.0f} hours, {:.0f} minutes".format(hours, minutes)
    elif minutes > 0:
        time_str = "{:.0f} minutes, {:.0f} seconds".format(minutes, seconds)
    else:
        time_str = "{:.2f} seconds".format(seconds)
    return time_str

timesteps = 1000000
path = "models/sac_tests_grid_search/{}_timesteps".format(timesteps)

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

l = 1.5  # separation between robots
ofst_x = (l/2)*((n_envs_x+1) % 2)
ofst_y = (l/2)*((n_envs_y+1) % 2)

starting_points = np.zeros((models.shape[0],models.shape[1],3))
for i in range(models.shape[0]):
    for j in range(models.shape[1]):
        starting_points[i][j][0] = ofst_x+l*math.ceil(0.5*j)*(-1)**j
        starting_points[i][j][1] = ofst_y+l*math.ceil(0.5*i)*(-1)**i
        starting_points[i][j][2] = 0


# start the pybullet server
server = PbEnvServer(gui=True)
title = "Trained: tau vs ent_coef ({})".format(steps2str(timesteps))
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

for i in range(2000):
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
