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
import pybullet_data


class Solo12Env(gym.Env):
    def __init__(self, 
                 healthy_reward=5.0,
                 terminate_when_unhealthy=True,
                 reset_noise_scale=0.1,
                 render_width=64,
                 render_height=64,
                 max_steps=500,
                 ):
        self.healthy_reward = healthy_reward
        self.step_counter = 0

        self.action_space = Box(low=-1.0, high=1.0, shape=(3,)) # vx, vy, vz
        self.observation_space = Dict({
            "state": Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64), # 12 joint positions, 6 imu readings
            "image": Box(low=0, high=255, shape=(render_height, render_width, 3), dtype=np.uint8)
        })
        
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.reset_noise_scale = reset_noise_scale
        self.render_width = render_width
        self.render_height = render_height
        self.max_steps = max_steps

        aspect = render_width / render_height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=aspect,
            nearVal=0.01,
            farVal=100
        )
        self.device = PyBulletSimulator()
        self.params = RLParams()
        self.device.Init(calibrateEncoders=True,
                      q_init=self.params.q_init,
                      envID=0,
                      use_flat_plane=True,
                      enable_pyb_GUI=False,
                      dt=self.params.dt,
                      alpha=self.params.alpha)
        
        self.robotId = self.device.pyb_sim.robotId
        
        self._initialize_policy()
        self._initialize_dynamic_objects()

        

    def _initialize_policy(self):
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

    def _initialize_dynamic_objects(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        mesh_scale = [0.1, 0.1, 0.1]
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName="sphere_smooth.obj",
            halfExtents=[0.5, 0.5, 0.1],
            rgbaColor=[1.0, 0.0, 0.0, 1.0],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0.0, 0.0, 0.0],
            meshScale=mesh_scale,
        )

        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="sphere_smooth.obj",
            collisionFramePosition=[0.0, 0.0, 0.0],
            meshScale=mesh_scale,
        )

        self.sphereId1 = p.createMultiBody(
            baseMass=0.4,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[1.6, 0.0, 0.1],
            useMaximalCoordinates=True,
        )

    def _controller(self, action):
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

    def _check_collision(self):
        contact_points = p.getContactPoints(self.robotId, self.sphereId1)
        return len(contact_points) > 0
    
    def _terminated(self):
        collided_with_sphere = self._check_collision()
        fallen_over = self.device.baseState[0][2] < 0.1

        return collided_with_sphere or fallen_over
    
    def _calculate_reward(self):
        if self._terminated():
            return 0.0
        distance_to_origin = np.linalg.norm(self.device.baseState[0][:2])
        return self.healthy_reward - distance_to_origin

    def _get_obs(self):
        obs = {}
        obs["state"] = np.concatenate([
            self.device.joints.positions,
            self.device.imu.accelerometer,
            self.device.imu.gyroscope
        ])
        obs["image"] = self.render()
        return obs

    def step(self, action):
        self._controller(action)
        obs = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._terminated()
        info = {}
        info["state"] = obs["state"]
        info["is_terminal"] = terminated
        info["base_link_pos"] = self.device.baseState[0]
        info["reward"] = reward
        info["step"] = self.step_counter

        if self.step_counter % 100 == 0:
            self._reset_and_apply_force_projectile()

        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            terminated = True

        return obs, reward, terminated, info
        
        
    def render(self):
        roll, pitch, yaw = doge_utils.quat_to_euler(self.device.baseState[1]) * 180 / math.pi
        cam_target_pos = self.device.baseState[0]
        cam_target_pos = (cam_target_pos[0], cam_target_pos[1], cam_target_pos[2] + 0.2)
        up_axis_idx = 2

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target_pos,
            distance=0.1,
            yaw=yaw - 90,
            pitch=pitch,
            roll=roll,
            upAxisIndex=up_axis_idx
        )

        (_, _, px, _, _) = p.getCameraImage(
            width=self.render_width,
            height=self.render_height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK
        )
        np_img_arr = np.array(px)
        frame = np_img_arr[:, :, :3]
        return frame

    def close(self):
        self.device.Stop()

    def _reset_and_apply_force_projectile(self):
        y_noise = np.random.uniform(-1, 1)
        xy_offset = self.device.baseState[0][:2]
        p.resetBasePositionAndOrientation(self.sphereId1, [3.0 + xy_offset[0], y_noise + xy_offset[1], 0.1], [0, 0, 0, 1])
        p.resetBaseVelocity(self.sphereId1, linearVelocity=[-5.5, -(xy_offset[1] + y_noise) * 1.3, 3.0])

    def _reset_model(self):
        z_reset = 0.3
        p.resetBasePositionAndOrientation(self.robotId, [0.0, 0.0, z_reset], [0, 0, 0, 1])

        return self._get_obs()

    def reset(self):
        self.step_counter = 0
        obs = self._reset_model()
        return obs

if __name__ == "__main__":
    show_frame = False
    env = Solo12Env()
    start = time.time()
    for j in range(1000):
        obs, reward, terminated, info = env.step([0.0, 0.0, 0.0])
        if terminated:
            env.reset()
        if show_frame:
            frame = env.render()
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", bgr_frame)
            cv2.waitKey(1)
    end = time.time()
    print(f"Time taken: {end - start}")
    env.close()