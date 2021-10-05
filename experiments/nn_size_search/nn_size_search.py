

# RL Algorithms and envs
import os
import itertools
import psutil
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.utils.folders import delete_contents
from experiments.utils.rl_helpers import make_env

def nn_size_search(timesteps=int(1e7)):
    """Try diffrent net architectures

    :param timesteps: training timesteps for each point on hyperparameter grid
    """

    log_dir = "tf_log/{}_timesteps/".format(timesteps)
    os.makedirs(log_dir, exist_ok=True)
    if timesteps < 1e6:
        delete_contents(log_dir)

    # Create log dir
    model_dir = "models/{}_timesteps/".format(timesteps)
    os.makedirs(model_dir, exist_ok=True)

    test_parameters = [
        # Policy network layers
        [
            [64, 64],
            [343, 343],
            [64, 64, 64],
            [64, 343, 64],
        ],

        # Q-function layers
        [
            [64, 64],
            [343, 343],
            [64, 64, 64],
            [64, 343, 64],
        ],
    ]

    parm_test_set = tuple(itertools.product(*test_parameters))

    num_cpu = psutil.cpu_count() - 1
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    for param in parm_test_set:
        pi = param[0]
        qf = param[1]

        print(f"--- pi: {pi} qf: {qf} ---")
        policy_kwargs = dict(net_arch=dict(pi=pi, qf=qf))

        name = f"SAC_pi_{len(pi)}.{pi[1]}__qf_{len(qf)}.{qf[1]}"


        vec_env.reset()
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            ent_coef="auto_0.1",
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs
        )

        model.learn(
            timesteps,
            tb_log_name=name
        )

        name = os.path.join(model_dir, name+".zip")
        model.save(name)

if __name__ == "__main__":
    nn_size_search()
