import numpy as np

import gymnasium as gym
from gymnasium import utils
from gymnasium.spaces import Box, Dict
import torch, time, math, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from external.quadruped_rl.PyBulletSimulator import PyBulletSimulator
from external.quadruped_rl.Params import RLParams
from external.quadruped_rl.cpuMLP import Interface
from testing import doge_utils
from pybullet_utils import bullet_client
import pybullet as p
import cv2

class Solo12Env(gym.Env):
    def __init__(self, healthy_reward=100.0):
        self.healthy_reward = healthy_reward
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,))
        self.observation_space = Dict({
            "observation": Box(low=-np.inf, high=np.inf, shape=(12,)),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=(12,)),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=(12,))
        })
        self.device = PyBulletSimulator()
        self.params = RLParams()
        self.device.Init(calibrateEncoders=True,
                      q_init=self.params.q_init,
                      envID=0,
                      use_flat_plane=True,
                      enable_pyb_GUI=False,
                      dt=self.params.dt,
                      alpha=self.params.alpha)
        
        self.policy = Interface()
        polDirName = "external/quadruped_rl/tmp_checkpoints/sym_pose/energy/6cm/w2/"
        estDirName = "external/quadruped_rl/tmp_checkpoints/state_estimation/symmetric_state_estimator.txt"
        self.policy.initialize(polDirName, estDirName, self.params.q_init.copy())

        
        # Init Histories **********************************************
        self.device.parse_sensor_data()
        self.policy.pTarget12 = self.params.q_init.copy()
        self.policy.update_observation(
            self.device.joints.positions.reshape((-1, 1)),
            self.device.joints.velocities.reshape((-1, 1)),
            self.device.imu.attitude_euler.reshape((-1, 1)),
            self.device.imu.gyroscope.reshape((-1, 1)),
        )

        self.device.joints.set_position_gains(self.policy.P)
        self.device.joints.set_velocity_gains(self.policy.D)
        self.device.joints.set_desired_positions(self.policy.pTarget12)
        self.device.joints.set_desired_velocities(np.zeros((12,)))
        self.device.joints.set_torques(np.zeros((12,)))

        for j in range(int(self.params.control_dt / self.params.dt)):
            self.device.send_command_and_wait_end_of_cycle(False)
            self.device.parse_sensor_data()

    def step(self, action):
        self.policy.vel_command = np.array(action)
        self.policy.update_observation(
            self.device.joints.positions.reshape((-1, 1)),
            self.device.joints.velocities.reshape((-1, 1)),
            self.device.imu.attitude_euler.reshape((-1, 1)),
            self.device.imu.gyroscope.reshape((-1, 1)),
        )

        q_des = self.policy.forward()

        self.device.joints.set_position_gains(self.policy.P)
        self.device.joints.set_velocity_gains(self.policy.D)
        self.device.joints.set_desired_positions(q_des)
        self.device.joints.set_desired_velocities(np.zeros((12,)))
        self.device.joints.set_torques(np.zeros((12,)))

        for j in range(int(self.params.control_dt / self.params.dt)):
            self.device.parse_sensor_data()
            self.device.send_command_and_wait_end_of_cycle(False)
        
    def render(self):
        roll, pitch, yaw = doge_utils.quat_to_euler(self.device.baseState[1]) * 180 / math.pi
        cam_target_pos = self.device.baseState[0]
        cam_target_pos = (cam_target_pos[0], cam_target_pos[1], cam_target_pos[2] + 0.2)
        up_axis_inx = 2
        width = 320
        height = 200
        near_plane = 0.01
        far_plane = 100
        fov = 60

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target_pos,
            distance=0.1,
            yaw=yaw - 90,
            pitch=pitch,
            roll=roll,
            upAxisIndex=up_axis_inx
        )
        aspect = width / height

        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)

        (_, _, px, _, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        np_img_arr = np.array(px)
        frame = np_img_arr[:, :, :3]
        return frame

    def close(self):
        pass

    def reset_model(self):
        pass

    def reset(self):
        pass

if __name__ == "__main__":
    env = Solo12Env()
    env.reset()
    for _ in range(1000):
        env.step([1.0, 0.0, 0.0])
        env.render()
    env.close()