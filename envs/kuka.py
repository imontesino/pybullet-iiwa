import copy
import inspect
import math
import os
import IPython

import numpy as np
import pybullet as p
import pybullet_data

import PyKDL as kdl
from kdl_parser import urdf as kdlurdf
from .kdl_error_codes import error2name

class Kuka:

    def __init__(self,
                 urdfRootPath="urdf/",
                 timestep=0.01,
                 robotModel="model.urdf",
                 debug=False,
                 useKDL=False,
                 startingPosition=[0,0,0]):

        self.debug = debug
        self.debug_line_id = None
        self.urdfRootPath = urdfRootPath

        # Use KDL Solver for IK instead of pybullet
        self.useKDL = useKDL

        self.robotFile = os.path.join(self.urdfRootPath,robotModel)
        self.filename, fileExtension = os.path.splitext(self.robotFile)
        if fileExtension.lower() == ".urdf":
            self.kukaUid = p.loadURDF(self.robotFile)
        elif fileExtension.lower() == ".sdf":
            objects = p.loadSDF(self.robotFile)
            self.kukaUid = objects[0]
        else:
            raise ValueError("Robot file not in .urdf or .sdf format")

        ## Kinematic Parameters

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

        if self.useKDL:
            if fileExtension == ".urdf":
                self.init_kdl()
            else:
                raise AssertionError("KDL requires a .urdf file. Given: "+fileExtension)


        self.basePos = np.array(startingPosition)
        self.baseOrn = np.array([0.000000, 0.000000, 0.000000, 1.000000])
        self.baseCoM = np.array(p.getBasePositionAndOrientation(self.kukaUid)[0])

        self.reset()

    def __del__(self):
        p.removeBody(self.kukaUid)
        print("{} with id: {} was removed from the simulation.".format(self.filename, self.kukaUid))

    def init_kdl(self):
        jointInfos = [p.getJointInfo(self.kukaUid, i) for i in range(7)]
        jointsMinKDL = kdl.JntArray(7)
        jointsMaxKDL = kdl.JntArray(7)
        for i, info in enumerate(jointInfos):
            jointsMinKDL[i] = info[8]
            jointsMaxKDL[i] = info[9]

        root = 'lbr_iiwa_link_0'
        tip = 'lbr_iiwa_link_7'
        ok, tree = kdlurdf.treeFromFile(self.robotFile)

        self.chain = tree.getChain(root,tip)

        # Solvers
        self.vik = kdl.ChainIkSolverVel_pinv(self.chain)
        self.fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self.ik = kdl.ChainIkSolverPos_NR_JL(self.chain,
                                            jointsMinKDL,
                                            jointsMaxKDL,
                                            self.fk,
                                            self.vik)

    def reset(self):
        # This methods uses the CoM of the base as the origin
        p.resetBasePositionAndOrientation(self.kukaUid,
                                          self.basePos + self.baseCoM,
                                          self.baseOrn)

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
        pose = []
        state = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)

        # Link states are given in world coordinates
        pos = np.array(state[0]) - self.basePos
        orn = np.array(state[1]) - self.baseOrn
        euler = p.getEulerFromQuaternion(orn)

        pose.extend(list(pos))
        pose.extend(list(euler))

        return pose

    def getJointPositions(self):
        jointPositions = []
        for jointIndex in self.motorIndices:
            jointPositions.append(p.getJointState(self.kukaUid, jointIndex)[0])
        return np.array(jointPositions)

    def getJointVelocities(self):
        jointVelocities = []
        for jointIndex in self.motorIndices:
            jointVelocities.append(p.getJointState(self.kukaUid, jointIndex)[1])
        return np.array(jointVelocities)


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
        pos = np.array(position) + self.basePos
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

    def inverseKinematics(self, pos, orn):
        if self.useKDL:
            jointPoses = self.inverseKinematicsKDL(pos, orn)
        else:
            jointPoses = self.inverseKinematicsPB(pos, orn)

        return np.array(jointPoses)

    def inverseKinematicsPB(self, targetPos, targetOrn,
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

    def inverseKinematicsKDL(self, targetPos, targetOrn,
                            initialGuess=None):
        #TODO: Clean-up
        if initialGuess == None:
            initialGuess = self.initialJointPositions[:7]

        pos = kdl.Vector(targetPos[0],targetPos[1],targetPos[2])

        rot_mat = p.getMatrixFromQuaternion(targetOrn)
        rot_x = kdl.Vector(rot_mat[0], rot_mat[1], rot_mat[2])
        rot_y = kdl.Vector(rot_mat[3], rot_mat[4], rot_mat[5])
        rot_z = kdl.Vector(rot_mat[6], rot_mat[7], rot_mat[8])
        orn = kdl.Rotation(rot_x, rot_y, rot_z)
        desiredFrame = kdl.Frame(orn, pos)

        q_init = kdl.JntArray(7)
        for i, q in enumerate(initialGuess):
            q_init[i] = q

        jointPoses = kdl.JntArray(7)
        ret = self.ik.CartToJnt(q_init,desiredFrame,jointPoses)

        if ret < 0:
            print("IK failed with error: ", error2name[ret])

        return list(jointPoses)

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

