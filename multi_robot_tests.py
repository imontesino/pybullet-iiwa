"""Este es el docstring del módulo"""

import math
import os
import time

from stable_baselines import DDPG, SAC, TD3, TRPO

from envs.kukaEnvs import pb_env_server

def steps2str(steps):
    """Convert traning steps to elapsed sim time

    :param steps: number of training steps simulated
    """
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

ALG_PATH = "models/algorithm_tests_new"
directories = os.listdir(ALG_PATH)

models = {}
for directory in directories:
    save_point_time = int(directory.split("_")[0])

    models_save_point = os.path.join(ALG_PATH, directory)
    algorithms = os.listdir(models_save_point)

    models_at_save_point = []
    for algorithm in algorithms:
        model_file = os.path.join(models_save_point, algorithm)
        buffer = {"name": algorithm.split(".")[0], "file": model_file}
        models_at_save_point.append(buffer)

    models[save_point_time] = models_at_save_point

algorithms = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG, "TRPO": TRPO}

envs = []
n_envs = len(algorithms.keys())
L = 1.5  # separation between robots
ofst = (L/2)*((n_envs+1) % 2)
starting_points = [
    [0, ofst+L*math.ceil(0.5*i)*(-1)**i, 0] for i in range(n_envs)
]

# start the pybullet server
server = pb_env_server(gui=True)

for position in starting_points:
    env = server.spawn_env(start_position=position)  # 1 env == 1 robot
    env.reset()
    envs.append(env)

for save_point in sorted(models.keys()):
    title = "Trained: "+steps2str(save_point)
    server.onscreen_title(title)

    # Get models for algorithms saved at this timestep count
    agents = []
    names = []
    for model in models[save_point]:
        if model["name"] in algorithms:
            algorithm = algorithms[model["name"]]
            agent = algorithm.load(model["file"])
            agents.append(agent)
            names.append(model["name"])

    # initial observations for each agent
    obs = [None for _ in range(n_envs)]
    for i, agent in enumerate(agents):
        envs[i].onscreen_title(names[i])
        obs[i] = envs[i].reset()

    for i in range(3000):
        t_start = time.time()

        for j, agent in enumerate(agents):
            env = envs[j]
            action = agent.predict(obs[j], deterministic=True)
            obs[j], _, _, _ = env.step(action[0])

        while time.time()-t_start < 1/240:  # slow down to real time
            time.sleep(1/24000)
        server.step()

    for env in envs:
        env.reset()
