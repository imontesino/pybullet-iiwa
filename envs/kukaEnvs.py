import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from numpy.lib.npyio import savez

from .kuka import Kuka
from .utils.splines import BezierGeneratorCartesian, BezierGeneratorSpherical, baseline_circle_trajectory

class PCEnv(gym.Env):
    """Position Control Environment for Kuka IIWA

    actions: 7-D vector of motor torque normalized to [-1,1]
    observation: Vector of Joint Values normalized to [-1,1]
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                episode_timesteps=10000,
                start_position=[0,0,0],
                gui=False,
                gui_height=480,
                gui_width=640,
                serverless=False,
                rand_traj_training=False,
                useKDL=True,
                reward_function="joint"):

        super(PCEnv, self).__init__()
        # Debug title text
        self.title_id = None

        # allow access to pybullet directly
        self.p = p

        # set the trajectory function
        self.rand_traj_training = rand_traj_training
        if not self.rand_traj_training:
            self.trajectory = baseline_circle_trajectory
        else:
            self.curve = BezierGeneratorSpherical(n_samples=5,
                                         duration=(episode_timesteps-200)) # still for last 1000
            self.trajectory = self.curve.evaluate

        # Select reward function
        reward_functions = {
            "joint": self.__joint_reward,
            "cartesian": self.__distance_reward
        }
        reward_options = list(reward_functions.keys())

        if reward_function in reward_options:
            self.__reward = reward_functions[reward_function]
        else:
            options_string = ", ".join(reward_options[:-1])+", or "+reward_options[-1]
            raise ValueError("'reward_function' argument must be "+options_string)


        # Target position viz id
        self.target_id = None

        # current timestep
        self.t = 0


        self.serverless = serverless

        self.gui = gui

        if not self.serverless:
            # start a pybullet server if not launched in serverless mode
            self.__start_pybullet(gui, gui_height, gui_width)

        if self.gui:
            self.draw_target(self.trajectory(0), new_id=True)

        self.episode_timesteps = episode_timesteps

        # Initialize a robot instance
        self.start_position = np.array(start_position)
        self.robot = Kuka(useKDL=useKDL, startingPosition=self.start_position)

        # Define action and observation space
        # They must be gym.spaces objects
        # There are 7 continous motor controls:
        self.action_space = spaces.Box(-1, 1, self.robot.getActionShape())

        # Observation space includescurent position and desired target
        # Each position is 6 dimensional:
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(self.observation()),))

    def __start_pybullet(self, gui, gui_height, gui_width):
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

    def step(self, action):
        done = False

        # Set control torques
        self.robot.torqueControl(action)

        # Step the simulation
        if not self.serverless:
            p.stepSimulation()

        self.t += 1

        # get achieved position and desired position
        observation = self.observation()

        # draw target
        if self.gui:
            self.draw_target(self.trajectory(self.t))

        # get reward based on distance
        reward = self.reward_function()

        info = {
            'is_success': done,
        }

        if(self.t >= self.episode_timesteps):
            done = True

        return observation, reward, done, info

    def reward_function(self):
        return self.__reward()

    def __distance_reward(self):
        # Minimize cartesian distance
        current_position = self.robot.getEndEffectorPose()
        target_position = self.trajectory(self.t)
        lin_distance = np.linalg.norm(
            (current_position[:3]-target_position[:3]))
        angle_distance = np.linalg.norm(
            (current_position[3:]-target_position[3:]))
        return - (lin_distance+angle_distance/(2*np.pi))/100

    def __joint_reward(self):
        # Minimize Joint Distance
        currJPos = self.robot.getJointPositions()
        currGoal = self.trajectory(self.t)
        goalJPos = self.robot.inverseKinematics(currGoal[0:3],
                                                p.getQuaternionFromEuler(currGoal[3:]))
        return - np.linalg.norm(currJPos-goalJPos)

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
        return np.array(obs)

    def stabilize(self, timesteps=200):
        # Start at begining of trajectory
        for i in range(timesteps):
            self.robot.positionControl(self.trajectory(self.t)[:3])
            if not self.serverless:
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
        """Generates a dataset (dict) in the format expected by ExprtDataset

        Parameters:
        traj (list): a list of observation collected from an agent episode

        Returns:
        dataset (dict): a dataset in the format expected by ExprtDataset.
        Contains the actions taken by an expert.

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

        nj = self.robot.numJoints

        for i in range(len(traj)):
            #curr_pos = traj[i][0:3]
            #curr_orn = traj[i][3:6]
            goal_pos = traj[i][6:9]
            goal_orn = traj[i][9:12]
            joint_positions = traj[i][12:12+nj] # 7 joints + grippers
            joint_velocities = traj[i][12+nj:12+2*nj]

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
                                                         debug=False)
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
        if not self.serverless:
            del self.robot
            p.disconnect(physicsClientId=self.physicsClient)

    def onscreen_title(self, text,
                       position=None,
                       orientation=[np.pi/2,0,np.pi/2],
                       color=[0.000000, 0.000000, 0.000000],# black
                       text_size=0.4):
        orientation=p.getQuaternionFromEuler(orientation)
        if position is None:
            position = self.start_position + np.array([0,0,0.9])
        position[1]=position[1]-len(text)*(0.09*text_size/0.4) # center text

        if self.title_id is None:
            self.title_id = p.addUserDebugText(text,
                                                position,
                                                color,
                                                textSize=text_size,
                                                textOrientation=orientation)
        else:
            p.addUserDebugText(text,
                                position,
                                color,
                                textSize=text_size,
                                replaceItemUniqueId=self.title_id,
                                textOrientation=orientation)

    def draw_target(self, target, new_id=False):
        start_p = target[0:3]
        orn_q= p.getQuaternionFromEuler(target[3:6])
        orn_mat = np.reshape(p.getMatrixFromQuaternion(orn_q), (3,3)) # pb returns (9,) vector
        end_p = start_p + orn_mat.dot(np.array([0,0,1]))*0.1

        if new_id:
            self.target_id=p.addUserDebugLine(start_p, end_p,
                                              lineWidth=5,
                                              lineColorRGB=[1,0,0])
        else:
            p.addUserDebugLine(start_p, end_p,
                                lineWidth=5,
                                lineColorRGB=[1,0,0],
                                replaceItemUniqueId=self.target_id)

    def set_realtime(self, value):
        if not self.serverless:
            p.setRealTimeSimulation(value)


class pb_env_server():

    def __init__(self, episode_timesteps=10000, gui=False, gui_height=480, gui_width=640):
        # Debug title text
        self.title_id = None

        # current timestep
        self.t = 0

        # handle of spawned envs
        self.envs = []

        self.__start_pybullet(gui, gui_height, gui_width)

        self._p = p

    def __del__(self):
        for i in range(p.getNumBodies()):
            p.removeBody(i)
        p.disconnect()


    def __start_pybullet(self, gui, gui_height, gui_width):
        # GUI visualizer prameters
        self.gui = False
        self.gui_width = gui_width
        self.gui_height = gui_height
        self.camera_data = [None]*12
        self.camera_data[10] = 3.2 # distance
        self.camera_data[9] = -3.8 # pitch
        self.camera_data[8] = 90 # yaw
        self.camera_data[11] = [0.0, 0.0, 1] #target

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

    def spawn_env(self, start_position=[0,0,0]):
        env=PCEnv(start_position=start_position, serverless=True)
        self.envs.append(env)
        return env

    def set_realtime(self, value):
        p.setRealTimeSimulation(value)

    def step(self):
        p.stepSimulation()

    def onscreen_title(self, text,
                    position=None,
                    orientation=[np.pi/2,0,np.pi/2],
                    color=[0.0, 0.0, 0.0],# black
                    text_size=0.6):
        if position is None:
            position = [0, 0, 2]
        position[1]=position[1]-len(text)*0.12 # center text
        orientation=p.getQuaternionFromEuler(orientation)
        if self.title_id is None:
            self.title_id = p.addUserDebugText(text,
                                                position,
                                                color,
                                                textSize=text_size,
                                                textOrientation=orientation)
        else:
            p.addUserDebugText(text,
                                position,
                                color,
                                textSize=text_size,
                                replaceItemUniqueId=self.title_id,
                                textOrientation=orientation)
