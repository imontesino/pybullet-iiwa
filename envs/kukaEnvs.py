import multiprocessing

import gym
import numpy as np
from numpy.lib.npyio import savez
import pybullet as p
import pybullet_data
from gym import spaces

from .kuka import Kuka
import time

class PCEnv(gym.Env):
    """Position Control Environment for Kuka IIWA

    actions: 7-D vector of motor torque normalized to [-1,1]
    observation: Vector of Joint Values normalized to [-1,1]
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, episode_timesteps=10000, gui=False, gui_height=480, gui_width=640):
        super(PCEnv, self).__init__()
        # Debug title text
        self.title_id = None

        # GUI visualizer prameters
        self.gui = False
        self.gui_width = gui_width
        self.gui_height = gui_height
        self.camera_data = [None]*12
        self.camera_data[10] = 1.8 # distance
        self.camera_data[9] = -35 # pitch
        self.camera_data[8] = 50 # yaw
        self.camera_data[11] = [0.0, 0.0, 0.3] #target

        optionstring = '--width={} --height={}'.format(
        self.gui_width, self.gui_height)

        # Start a pybullet instance
        if gui:
            self.physicsClient = p.connect(p.GUI, options=optionstring)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            camera_data = list(p.getDebugVisualizerCamera())

            dist = self.camera_data[10]
            pitch = self.camera_data[9]
            yaw = self.camera_data[8]
            target = self.camera_data[11]

            p.resetDebugVisualizerCamera(dist, yaw, pitch, target)
            self.gui = True
        else:
            self.physicsClient = p.connect(p.DIRECT, options=optionstring)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.79983)
        self.planeId = p.loadURDF("plane.urdf")
        self.timestep = 1/240

        # current timestep
        self.t = 0

        self.episode_timesteps = episode_timesteps

        # Initialize a robot instance
        self.robot = Kuka()

        # Define action and observation space
        # They must be gym.spaces objects
        # There are 7 continous motor controls:
        self.action_space = spaces.Box(-1, 1, self.robot.getActionShape())

        # Observation space includescurent position and desired target
        # Each position is 6 dimensional:
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(self.observation()),))

    def trajectory(self, t):
        # TODO make this editable online
        radius = 0.1
        [0.537, 0.0, 0.5]
        x = 0.5321173259355029
        y = -0.0011066349287836563
        z = 0.55229843834858136
        x += 0
        y += np.sin(2*t*2*np.pi/5000)*radius
        z += (np.cos(2*t*2*np.pi/5000)-1)*radius
        return np.array([x, y, z, 0, -np.pi, 0])

    def step(self, action):
        done = False

        # Set control torques
        self.robot.torqueControl(action)

        # Step the simulation
        p.stepSimulation()
        self.t += 1

        # get achieved position and desired position
        observation = self.observation()

        # get reward based on distance
        reward = self.reward_function()

        info = {
            'is_success': done,
        }

        if(self.t >= self.episode_timesteps):
            done = True

        return observation, reward, done, info

    def reward_function(self):
        # Minimize the distance
        current_position = self.robot.getEndEffectorPose()
        target_position = self.trajectory(self.t)
        lin_distance = np.linalg.norm(
            (current_position[:3]-target_position[:3]))
        angle_distance = np.linalg.norm(
            (current_position[3:]-target_position[3:]))
        return - 10*(lin_distance+angle_distance/(2*np.pi))

    def reset(self):
        self.robot.reset()
        self.t = 0
        # get achieved position and desired position
        return self.observation()  # reward, done, info can't be included

    def observation(self):
        obs = []
        current_position = self.robot.getEndEffectorPose()
        target_position = self.trajectory(self.t)
        obs.extend(list(current_position))
        obs.extend(list(target_position))
        obs.extend(list(self.robot.getJointPositions()))
        obs.extend(list(self.robot.getJointVelocities()))
        return obs

    def stabilize(self, timesteps=200):
        # Start at begining of trajectory
        for i in range(timesteps):
            self.robot.positionControl(self.trajectory(self.t)[:3])
            p.stepSimulation()

    def generate_expert_traj(self, num_episodes=1, episode_timesteps=None, verbose=0):
        """Generate a dataset from the actions taken by an expert agent (PID+IK)

        Parameters:
        num_episodes (int): Number of episodes for which to record the expert
        agent

        Returns:
        dict: Dataset in the format required by the pretraining method

        """
        if episode_timesteps is None:
            episode_timesteps = self.episode_timesteps

        # The expert dataset is saved in python dictionary format with
        # keys: actions, episode_returns, rewards, obs, episode_starts.

        # obs, actions: shape (N * L, S)
        # where N = # episodes, L = episode length and S is the environment observation/action
        # space.

        actions = np.zeros(
            (num_episodes*episode_timesteps, 7), dtype=float)
        episode_returns = np.zeros((num_episodes,), dtype=float)
        rewards = np.zeros((num_episodes*episode_timesteps,), dtype=float)
        obs = np.zeros((num_episodes*episode_timesteps,
                       len(self.observation())), dtype=float)
        episode_starts = np.zeros(
            (num_episodes*episode_timesteps,), dtype=bool)

        self.stabilize()

        for n in range(num_episodes):
            t_start = time.time()
            # Mark episode beginning with True for the corresponding element in the array
            episode_starts[n*episode_timesteps] = True

            for i in range(episode_timesteps):
                # Get torques applied by the position controller
                actions[i] = self.robot.getTorquesFromTarget(
                    self.trajectory(i)[:3])

                # Save the observation an the reward
                obs[i], rewards[i], _, _ = self.step(actions[i])

            # Total rewards in an episode
            episode_returns[n] = rewards[
                n*episode_timesteps:(n+1)*episode_timesteps
            ].sum()

            self.t=0

            t_end = time.time()

            if verbose > 0:
                print("--- Episode {} Finished ---".format(n))
                print("episode_returns = {}, time = {:0.2f}s".format(episode_returns[n], t_end-t_start))
                print()

        # Disable the position controller after using it in
        # getTorquesFromTrajectory
        self.robot.enableTorqueControl()
        self.robot.eraseDebugLines()

        dataset = {
            "actions": actions,
            "episode_returns": episode_returns,
            "rewards": rewards,
            "obs": obs,
            "episode_starts": episode_starts
        }

        return dataset

    def generate_expert_from_traj(self, traj):
        """Summary or Description of the Function

        Parameters:
        argument1 (int): Description of arg1

        Returns:
        int:Returning value

        """
        # The expert dataset is saved in python dictionary format with
        # keys: actions, episode_returns, rewards, obs, episode_starts.

        # obs, actions: shape (N * L, S)
        # where N = # episodes, L = episode length and S is the environment observation/action
        # space.
        actions = np.zeros((len(traj), 7), dtype=float)
        episode_returns = np.zeros((1,), dtype=float)
        rewards = np.zeros((len(traj),), dtype=float)
        obs = np.zeros((len(traj), len(self.observation())), dtype=float)
        episode_starts = np.zeros((len(traj),), dtype=bool)


        # Mark episode beginning with True for the corresponding element in the array
        episode_starts[0] = True

        t_joint = np.zeros(len(traj))
        t_torques = np.zeros(len(traj))

        for i in range(len(traj)):
            #curr_pos = traj[i][0:3]
            #curr_orn = traj[i][3:6]
            goal_pos = traj[i][6:9]
            goal_orn = traj[i][9:12]
            joint_positions = traj[i][12:24] # 7 joints + grippers
            joint_velocities = traj[i][24:36]

            t_joint_start = time.time()
            # Set the robot to the position in trajectory
            self.robot.setJointStates(joint_positions,
                                      jointVelocities=joint_velocities)
            t_joint_end = time.time()


            t_torque_start = time.time()
            # Get torques applied by the position controller for the goal at
            # the time of the trajectory
            actions[i] = self.robot.getTorquesFromTarget(goal_pos,
                                                         orn=goal_orn,
                                                         debug=True)
            t_torque_end = time.time()

            t_joint[i] = t_joint_end - t_joint_start
            t_torques[i] = t_torque_end - t_torque_start

            # Compute reward
            rewards[i] = self.reward_function()

        # Total rewards in an episode
        episode_returns[0] = rewards.sum()

        # Disable the position controller after using it in
        # getTorquesFromTrajectory
        self.robot.enableTorqueControl()
        self.robot.eraseDebugLines()

        dataset = {
            "actions": actions,
            "episode_returns": episode_returns,
            "rewards": rewards,
            "obs": obs,
            "episode_starts": episode_starts
        }

        return dataset

    def render(self, mode='human', shadow=0):

        if mode == "human":
            print("Not yet implemented")
        elif mode == "rgb_array":
            base_pos,orn = p.getBasePositionAndOrientation(
                self.robot.kukaUid)

            dist = self.camera_data[10]
            pitch = self.camera_data[9]
            yaw = self.camera_data[8]
            target = self.camera_data[11]

            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target,
                distance=dist,
                yaw=yaw,
                pitch=pitch,
                roll=0,
                upAxisIndex=2)

            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self.gui_width)/self.gui_height,
                nearVal=0.1,
                farVal=100.0)

            (_, _, px, _, _) = p.getCameraImage(
                width=self.gui_width,
                height=self.gui_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL)

            rgb_array = np.array(px)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def close(self):
        del self.robot
        p.disconnect(physicsClientId=self.physicsClient)

    def onscreen_title(self, text,
                       position=[-0.000000, -1.0700, 0.9000],
                       color=[0.000000, 0.000000, 0.000000]): # black
        if self.gui:
            if self.title_id is None:
                self.title_id = p.addUserDebugText(text,
                                                   position,
                                                   color,
                                                   textSize=7,
                                                   replaceItemUniqueId=0)
            else:
                p.addUserDebugText(text,
                                   position,
                                   color,
                                   textSize=7,
                                   replaceItemUniqueId=0)

    def set_realtime(self, value):
        p.setRealTimeSimulation(value)
