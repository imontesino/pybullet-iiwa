import copy
import inspect
import math
import os

import numpy as np
import pybullet as p
import pybullet_data

class Kuka:

    def __init__(self,
                 urdfRootPath="urdf/",
                 timestep=0.01,
                 robotModel="kuka_with_gripper2.sdf",
                 debug=False):
        self.debug = debug
        self.debug_line_id = None
        self.urdfRootPath = urdfRootPath
        self.timestep = timestep
        self.maxVelocity = .35
        self.maxForce = 200.
        self.fingerAForce = 2
        self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 21
        self.useOrientation = 1
        self.kukaEndEffectorIndex = 6
        self.kukaGripperIndex = 7
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficients
        self.jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        # Initial configuration
        self.initialJointPositions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
            -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        ]

        self.basePos = [-0.100000, 0.000000, 0.070000]
        self.baseOrn = [0.000000, 0.000000, 0.000000, 1.000000]

        robotFile = os.path.join(self.urdfRootPath,robotModel)
        self.fileName, fileExtension = os.path.splitext(robotFile)
        if fileExtension.lower() == ".urdf":
            self.kukaUid = p.loadURDF(robotFile)
        elif fileExtension.lower() == ".sdf":
            objects = p.loadSDF(robotFile)
            self.kukaUid = objects[0]
        else:
            raise ValueError("Robot file not in .urdf or .sdf format")

        self.reset()

    def __del__(self):
        p.removeBody(self.kukaUid)
        print("{} with id: {} was removed from the simulation.".format(self.fileName, self.kukaUid))

    def reset(self):
        # for i in range (p.getNumJoints(self.kukaUid)):
        #  print(p.getJointInfo(self.kukaUid,i))
        p.resetBasePositionAndOrientation(self.kukaUid, self.basePos, self.baseOrn)

        self.numJoints = p.getNumJoints(self.kukaUid)
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex,
                              self.initialJointPositions[jointIndex])

        self.endEffectorPos = [0.537, 0.0, 0.5]
        self.endEffectorAngle = 0

        self.motorNames = []
        self.motorIndices = []

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.kukaUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                # print("motorname")
                # print(jointInfo[1])
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

        self.enableTorqueControl()

    def reset_zero(self):
        p.resetBasePositionAndOrientation(self.kukaUid, [-0.100000, 0.000000, 0.070000],
                                          [0.000000, 0.000000, 0.000000, 1.000000])

        self.numJoints = p.getNumJoints(self.kukaUid)
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex,0.0)

    # TODO no references to RL in robot
    def getActionShape(self):
        return (7,)  # A torque for each motor

    def getEndEffectorPose(self):
        """The current position and orientation of the end effector
        """
        observation = []
        state = p.getLinkState(self.kukaUid, self.kukaGripperIndex)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    def getJointPositions(self):
        jointPositions = []
        for jointIndex in self.motorIndices:
            jointPositions.append(p.getJointState(self.kukaUid, jointIndex)[0])
        return jointPositions

    def getJointVelocities(self):
        jointVelocities = []
        for jointIndex in self.motorIndices:
            jointVelocities.append(p.getJointState(self.kukaUid, jointIndex)[1])
        return jointVelocities


    def setJointStates(self, jointPositions,
                       jointVelocities=None):
        if jointVelocities is None:
            jointVelocities = [0]*len(jointPositions)

        for jointIndex in range(len(jointPositions)):
            p.resetJointState(self.kukaUid,
                              jointIndex,
                              jointPositions[jointIndex],
                              targetVelocity=jointVelocities[jointIndex])

    # TODO no references to RL in robot
    def applyAction(self, motorCommands):
        self.torqueControl(motorCommands)

    def positionControl(self, position,
                        orn=[0, -math.pi, 0],
                        initalGuess=None):
        pos = position
        orn = p.getQuaternionFromEuler(orn)

        jointPoses = self.inverseKinematics(pos, orn)

        p.setJointMotorControlArray(self.kukaUid,
                                    jointIndices=range(
                                        self.kukaEndEffectorIndex + 1),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=jointPoses[:self.kukaEndEffectorIndex + 1],
                                    targetVelocities=[0 for i in range(
                                        self.kukaEndEffectorIndex + 1)],
                                    forces=[self.maxForce for i in range(
                                        self.kukaEndEffectorIndex + 1)],
                                    positionGains=[0.3 for i in range(
                                        self.kukaEndEffectorIndex + 1)],
                                    velocityGains=[1 for i in range(self.kukaEndEffectorIndex + 1)])

    def inverseKinematics(self, targetPos, targetOrn,
                          initialGuess=None):
        if initialGuess is None:
            jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                    self.kukaEndEffectorIndex,
                                                    targetPos,
                                                    targetOrn,
                                                    jointDamping=self.jd)
        else:
            jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                    self.kukaEndEffectorIndex,
                                                    targetPos,
                                                    targetOrn,
                                                    lowerLimits=self.ll,
                                                    upperLimits=self.ul,
                                                    jointRanges=self.jr,
                                                    restPoses=initialGuess,
                                                    currentPosition=initialGuess,
                                                    jointDamping=self.jd)

        return jointPoses

    def enableTorqueControl(self):
        jointFriction = 0.01
        p.setJointMotorControlArray(self.kukaUid,
                                    jointIndices=range(
                                        self.kukaEndEffectorIndex + 1),  # 7
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[jointFriction]*7)

    def torqueControl(self, torques_norm):
        torques = np.array(torques_norm)*self.maxForce
        p.setJointMotorControlArray(self.kukaUid,
                                    jointIndices=range(
                                        self.kukaEndEffectorIndex + 1),  # 7
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=torques)

    def jointControl(self, jointPoses):

        for i in range(self.kukaEndEffectorIndex + 1):
            p.setJointMotorControl2(bodyUniqueId=self.kukaUid,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    targetVelocity=0,
                                    force=self.maxForce,
                                    maxVelocity=self.maxVelocity,
                                    positionGain=0.3,
                                    velocityGain=1)

    def getTorquesFromTarget(self, position, orn=[0, -math.pi, 0], debug=False):
        if debug:
            end_eff_pos = self.getEndEffectorPose()[:3]
            if self.debug_line_id is None:
                self.debug_line_id = p.addUserDebugLine(end_eff_pos, position,
                                                        lineWidth=5,
                                                        lineColorRGB=[1,0,0]
                                                        )
            else:
                p.addUserDebugLine(end_eff_pos, position,
                                   lineWidth=5,
                                   lineColorRGB=[1,0,0],
                                   replaceItemUniqueId=self.debug_line_id)

        self.positionControl(position, orn=orn, initalGuess=self.initialJointPositions)

        # Step the simulation to compute the torques applied by the position
        # controller
        p.stepSimulation()

        jointStates = p.getJointStates(self.kukaUid, jointIndices=range(self.kukaEndEffectorIndex + 1))

        # torque is the 4th element in each joinState tuple
        jointTorques = np.array([state[3] for state in jointStates])

        return jointTorques/self.maxForce

    def eraseDebugLines(self):
        if not self.debug_line_id is None:
            p.removeUserDebugItem(self.debug_line_id)
            self.debug_line_id = None

