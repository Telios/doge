import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
import torch, esim_torch, time
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from testing.mujoco_object import MujocoObject
from testing import doge_utils
from external.IEBCS.src.event_buffer import EventBuffer
from external.IEBCS.src.dvs_sensor import init_bgn_hist_cpp
from external.IEBCS.src.event_display import EventDisplay
import dsi



DEFAULT_CAMERA_CONFIG = {
    "distance": 0.75,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

class Solo12Env(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
            self,
            xml_file="/home/ubuntu/victor/learning_world_model/gym/envs/assets/scene.xml",
            ctrl_cost_weight=0.5,
            use_contact_forces=False,
            contact_cost_weight=5e-4,
            healthy_reward=100.0,
            terminate_when_unhealthy=True, # default True
            healthy_z_range=(-0.3, 0.1),
            goal_z = -0.139,
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            use_last_action=True,
            use_imu_data=True,
            use_command=True,
            **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            goal_z,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            use_last_action,
            use_imu_data,
            use_command,
            **kwargs,
            )
        
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._goal_z = goal_z
        self._time_limit = 24
        self._time_reset = 2
        self._time_elapsed = 0
        self._time_start_ns = time.time_ns()

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._is_init = False
        
        self._home_position = np.array([0.157, 0.22, -1.0, -0.157, 0.22, -1.0, 0.157, -0.22, 1.0, -0.157, -0.22, 1.0])
        
        self._action_damping = 0.3
        
        self._scales = {
            "joint_torque": -2.5e-2,
            "joint_accel": -2.5e-7,
            "action_rate": -0.1,
            "feet_air_time": 0.2,
            "foot_slip": -0.1,
            "velocity": -0.5,
            "heading": -0.1,
            "tracking_sigma": 0.25,
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.8,
            "lin_vel_z": -2.0,
            "ang_vel_z": -0.05,
            "orientation": -5.0,
            "stand_still": -0.5,
            "termination": -1.0,
        }
        
        self._state = {
            "base_link_pos": np.zeros(3),
            "base_link_quat": np.zeros(4),
            "base_link_euler": np.zeros(3),
            "FL_SHOULDER_pos": np.zeros(3),
            "FR_SHOULDER_pos": np.zeros(3),
            "HL_SHOULDER_pos": np.zeros(3),
            "HR_SHOULDER_pos": np.zeros(3),
            "last_action": np.zeros(12),
            "torques": np.zeros(12),
            "imu": np.zeros(6),
            "command": np.zeros(3),
        }

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces
        
        self._use_last_action = use_last_action
        
        self._use_imu_data = use_imu_data

        self._use_command = use_command
        
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        obs_shape = 19
        if not self._exclude_current_positions_from_observation:
            obs_shape += 6
        if use_contact_forces:
            obs_shape += 12
        if use_last_action:
            obs_shape += 12
        if use_imu_data:
            obs_shape += 6
        if use_command:
            obs_shape += 3
        
        observation_space = Dict(
            {"state":
            Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape,),
            dtype=np.float64,
        ),
            }
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            camera_id=0,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )
    
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
    
    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces
    
    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(np.square(self.contact_forces))
        return contact_cost
    
    @property
    def is_healthy(self):
        state = self.get_body_com("base_link")
        min_z, max_z = self._healthy_z_range
        # determine if upright using euler angles
        base_link_euler = self._state["base_link_euler"]
        roll = base_link_euler[1]
        pitch = base_link_euler[2]
        is_upright = np.abs(pitch) < 0.8 and np.abs(roll) < 0.8
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z and is_upright
        return is_healthy
    
    @property
    def terminated(self):
        healthy = not self.is_healthy if self._terminate_when_unhealthy else False
        time_up = self._time_elapsed >= self._time_limit
        return healthy or time_up
    
    def sample_command(self):
        lin_vel_x = [-0.6, 1.5] # min max [m/s]
        lin_vel_y = [-0.8, 0.8] # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7] # min max [rad/s]
        
        lin_vel_x = np.random.uniform(lin_vel_x[0], lin_vel_x[1])
        
        lin_vel_y = np.random.uniform(lin_vel_y[0], lin_vel_y[1])
        
        ang_vel_yaw = np.random.uniform(ang_vel_yaw[0], ang_vel_yaw[1])
        
        new_command = np.array([lin_vel_x, lin_vel_y, ang_vel_yaw])
        return new_command
        
    def update_state(self):
        self._state["base_link_pos"] = self.get_body_com("base_link")[:3]
        self._state["base_link_quat"] = self.data.body("base_link").xquat
        self._state["base_link_euler"] = doge_utils.quat_to_euler(self._state["base_link_quat"])
        self._state["FL_SHOULDER_pos"] = self.get_body_com("FL_SHOULDER")[:3]
        self._state["FR_SHOULDER_pos"] = self.get_body_com("FR_SHOULDER")[:3]
        self._state["HL_SHOULDER_pos"] = self.get_body_com("HL_SHOULDER")[:3]
        self._state["HR_SHOULDER_pos"] = self.get_body_com("HR_SHOULDER")[:3]
        self._state["torques"] = self.data.qfrc_actuator.flat.copy()
        self._state["imu"] = self.data.sensordata
        self._state["base_link_vel"] = self.get_velocity_of_body("base_link")
        
    def step(self, action):
      damped_action = self._home_position + action * self._action_damping
      self.do_simulation(damped_action, self.frame_skip)
      self.update_state()
      reward = self.calculate_reward(damped_action)
      
      self._time_elapsed += self.dt

      terminated = self.terminated
      observation = self._get_obs()
      info = {
            "reward": reward,
            "is_terminal": terminated,
            "time_elapsed": self._time_elapsed,
        }
      
      self._state["last_action"] = damped_action
      
      return observation, reward, terminated, info  
    
    def calculate_reward(self, action):
        healthy_reward = self.healthy_reward * self.dt
        z_goal_reward = 100 - np.abs(self._state["base_link_pos"][2] - self._goal_z)
        - np.abs(self._state["FR_SHOULDER_pos"][2] - self._goal_z) 
        - np.abs(self._state["FL_SHOULDER_pos"][2] - self._goal_z)
        - np.abs(self._state["HL_SHOULDER_pos"][2] - self._goal_z)
        - np.abs(self._state["HR_SHOULDER_pos"][2] - self._goal_z) 
        z_goal_reward *= self.dt 
        
        lin_vel_err = np.sum(np.abs(self._state["command"][:2] - self._state["base_link_vel"][:2]))
        lin_vel_reward = np.exp(-lin_vel_err / self._scales["tracking_sigma"]) * self.dt
        
        ang_vel_err = np.abs(self._state["command"][2] - self._state["base_link_vel"][5])
        ang_vel_reward = np.exp(-ang_vel_err / self._scales["tracking_sigma"]) * self.dt
        
        
        torque_regulizer = np.sqrt(np.sum(np.square(self._state["torques"]))) + np.sum(np.abs(self._state["torques"]))
        action_regulizer = np.sum(np.square(action - self._state["last_action"]))
        velocity_regulizer = np.sum(np.square(self._state["base_link_vel"]))
        
        torque_regulizer *= self.dt
        action_regulizer *= self.dt
        velocity_regulizer *= self.dt
        
        reward = z_goal_reward + self._scales["tracking_lin_vel"] * lin_vel_reward + self._scales["tracking_ang_vel"] * ang_vel_reward + healthy_reward + self._scales["joint_torque"] * torque_regulizer + self._scales["action_rate"] * action_regulizer + self._scales["velocity"] * velocity_regulizer
        reward = np.clip(reward, 0.0, 10000.0)
        return reward
   

    def check_collision(self):
        # get ids for all solo12 geometries
        body_names = {'base_link', 'FL_SHOULDER', 'FL_UPPER_LEG', 'FL_LOWER_LEG', 'FL_FOOT', 'HL_SHOULDER', 'HL_UPPER_LEG', 'HL_LOWER_LEG', 'HL_FOOT', 'FR_SHOULDER', 'FR_UPPER_LEG', 'FR_LOWER_LEG', 'FR_FOOT', 'HR_SHOULDER', 'HR_UPPER_LEG', 'HR_LOWER_LEG', 'HR_FOOT'}
        body_ids = [self.data.body(name).id for name in body_names]
        ball_id = self.model.body('projectile1').id
        for contact in self.data.contact:
            if self.model.geom_bodyid[contact.geom[0]] in body_ids and self.model.geom_bodyid[contact.geom[1]] == ball_id or self.model.geom_bodyid[contact.geom[1]] in body_ids and self.model.geom_bodyid[contact.geom[0]] == ball_id:
                return True
        return False
    
    def get_velocity_of_body(self, body_name):
        id = self.data.body(body_name).id
        velocities = self.data.qvel[id : id + 6]
        return velocities

    def _get_obs(self):
        obs = {}
        obs["state"] = self.data.qpos[:-7].flat.copy() 
        if self._use_last_action:
            obs["state"] = np.concatenate([obs["state"], self._state["last_action"]])
        if self._use_imu_data:
            obs["state"] = np.concatenate([obs["state"], self._state["imu"]])
        if self._use_command:
            obs["state"] = np.concatenate([obs["state"], self._state["command"]])
        return obs
        
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos[:-7] + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq - 7
        )
        qpos = np.concatenate([qpos, np.array([3.0, 0, 100.0, 1.0, 0.0, 0.0, 0.0])])
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        
        self._state["command"] = self.sample_command()
        
        observation = self._get_obs()
        
        return observation
    
    def reset(
        self,
    ):
        self._reset_simulation()
        ob = self.reset_model()
        self._time_elapsed = 0
        return ob
    
    def render(self, camera_id=0):
        return self.mujoco_renderer.render(render_mode=self.render_mode, camera_id=camera_id)
    