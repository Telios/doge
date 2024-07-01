import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
import torch, esim_torch, time
import cv2 as cv
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
            healthy_z_range=(-0.3, 0.2),
            goal_z = -0.2,
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            use_last_action=True,
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
        self._esim = esim_torch.ESIM(
            contrast_threshold_neg=0.2,
            contrast_threshold_pos=0.2,
            refractory_period_ns=0,
        )

        self._width = 64
        self._height = 64
        dsi.initSimu(self._height, self._width)
        dsi.initLatency(200, 50, 50, 300)
        dsi.initContrast(0.3, 0.3, 0.05)
        init_bgn_hist_cpp(f"{os.getcwd()}/../external/IEBCS/data/noise_pos_161lux.npy", f"{os.getcwd()}/../external/IEBCS/data/noise_pos_161lux.npy")
        self._ev_full = EventBuffer(1)
        self._ed = EventDisplay("Events", self._width, self._height, 2000)
        self._is_init = False
        
        self._scales = {
            "joint_torque": -2.5e-2,
            "joint_accel": -2.5e-7,
            "action_rate": -0.1,
            "feet_air_time": 0.2,
            "foot_slip": -0.1,
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
        }

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces
        
        self._use_last_action = use_last_action

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
        
        observation_space = Dict(
            {"state":
            Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape,),
            dtype=np.float64,
        ),
            "image":
            Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8,
            ),}
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

        self._projectile = MujocoObject(self.model, self.data, 'projectile1')

        self._force = doge_utils.calculate_force_vector(self.data.body('base_link').xpos, self._projectile.position())

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
        return healthy or time_up or self.check_collision()
    
    def update_state(self):
        self._state["base_link_pos"] = self.get_body_com("base_link")[:3]
        self._state["base_link_quat"] = self.data.body("base_link").xquat
        self._state["base_link_euler"] = doge_utils.quat_to_euler(self._state["base_link_quat"])
        self._state["FL_SHOULDER_pos"] = self.get_body_com("FL_SHOULDER")[:3]
        self._state["FR_SHOULDER_pos"] = self.get_body_com("FR_SHOULDER")[:3]
        self._state["HL_SHOULDER_pos"] = self.get_body_com("HL_SHOULDER")[:3]
        self._state["HR_SHOULDER_pos"] = self.get_body_com("HR_SHOULDER")[:3]
        self._state["torques"] = self.data.qfrc_actuator.flat.copy()
        
    def step(self, action):
      self.apply_force()
      self.do_simulation(action, self.frame_skip)
      self.update_state()
      reward = self.calculate_reward(action)
      
      timestamp_ns = time.time_ns() - self._time_start_ns
      self._time_elapsed += self.dt

      terminated = self.terminated
      observation = self._get_obs()
      # convert to event image
      #observation['image'] = self.get_event_image(observation['image'], timestamp_ns, mode="iebcs")
      info = {
            "reward": reward,
            "is_terminal": terminated,
            "time_elapsed": self._time_elapsed,
        }
      
      self._state["last_action"] = action

      # reset dynamic object after time_limit
      if self.data.time % self._time_reset < 0.02:
          # reset projectile to offset from base_link
          offset = self._projectile.initial_pos
          offset[0] += self._state["base_link_pos"][0]
          offset[1] += self._state["base_link_pos"][1]
          self._projectile.set_position_to(offset)
          self._projectile.reset_force()
          self._force = doge_utils.calculate_force_vector(self.data.body('base_link').xpos, self._projectile.position())
      
      return observation, reward, terminated, info  
    
    def calculate_reward(self, action):
        healthy_reward = self.healthy_reward * self.dt
        z_goal_reward = 100 - np.abs(self._state["base_link_pos"][2] - self._goal_z)
        - np.abs(self._state["FR_SHOULDER_pos"][2] - self._goal_z) 
        - np.abs(self._state["FL_SHOULDER_pos"][2] - self._goal_z)
        - np.abs(self._state["HL_SHOULDER_pos"][2] - self._goal_z)
        - np.abs(self._state["HR_SHOULDER_pos"][2] - self._goal_z) 
        z_goal_reward *= self.dt 
        distance_to_origin = np.linalg.norm(self._state["base_link_pos"][:2]) * self.dt
        
        torque_regulizer = np.sqrt(np.sum(np.square(self._state["torques"]))) + np.sum(np.abs(self._state["torques"]))
        action_regulizer = np.sum(np.square(action - self._state["last_action"]))
        
        torque_regulizer *= self.dt
        action_regulizer *= self.dt
        
        return z_goal_reward - distance_to_origin + healthy_reward + self._scales["joint_torque"] * torque_regulizer + self._scales["action_rate"] * action_regulizer
        
    def get_event_image(self, img_in, timestamp, mode="esim"):
        if mode == "esim":
            return self.get_event_image_esim(img_in, timestamp)
        elif mode == "iebcs":
            return self.get_event_image_iebcs(img_in)
        else:
            return self.get_event_image_esim(img_in, timestamp)
        
    def get_event_image_esim(self, img_in, timestamp):
        img = cv.cvtColor(img_in, cv.COLOR_RGB2GRAY)
        img = np.log(img.astype("float32") / 255 + 1e-4)
        img = torch.from_numpy(img).to(self.DEVICE)
        timestamp = torch.tensor([timestamp], device=self.DEVICE, dtype=torch.int64)

        events = self._esim.forward(img, torch.tensor([timestamp], device=self.DEVICE))
        if events is None:
            # return white image with the same size as the input image
            return np.ones((img_in.shape[0], img_in.shape[1], 3), dtype=np.uint8) * 255
        image_color = torch.stack([img, img, img], -1)
        image_color[:,:,:] = 255
        image_color[events['y'], events['x'], :] = 0
        image_color[events['y'], events['x'], events['p']] = 200

        image_color[:,:,0] = image_color[:,:,1]
        image_color = image_color.cpu().numpy()
        image_color = image_color.astype(np.uint8)
        return image_color
    
    def get_event_image_iebcs(self, img_in):
        dt = 1000
        if img_in is None:
            return
        img = cv.cvtColor(img_in, cv.COLOR_RGB2LUV)[:, :, 0]
        if not self._is_init:
            dsi.initImg(img)
            self._is_init = True
        else:
            buf = dsi.updateImg(img, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            100000000)
            self._ed.update(ev, dt)
            self._ev_full.increase_ev(ev)
        return self._ed.im

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

    def apply_force(self):
        if self.data.time % self._time_reset < 0.2:
            self._projectile.apply_force(self._force, self.model, self.data)
        elif self.data.time % self._time_reset > 0.3 and self.data.time % self._time_reset < 0.4:
            self.data.qfrc_applied = np.zeros(self.model.nv)

    def _get_obs(self):
        obs = {}
        obs["state"] = self.data.qpos[:-7].flat.copy()
        if self._use_last_action:
            obs["state"] = np.concatenate([obs["state"], self._state["last_action"]])
        obs["image"] = self.render(camera_id=1)
        #obs["image_color"] = obs["image"]
        #obs["overview_img"] = self.render(camera_id=1)
        return obs
        
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos[:-7] + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq - 7
        )
        qpos = np.append(qpos, self._projectile.get_random_reset_pos())
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        self._force = doge_utils.calculate_force_vector(self.data.body('base_link').xpos, self._projectile.position())
        
        observation = self._get_obs()
        timestamp_ns = time.time_ns() - self._time_start_ns
        #observation['image'] = self.get_event_image(observation['image'], timestamp_ns, mode="iebcs")
        
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
    