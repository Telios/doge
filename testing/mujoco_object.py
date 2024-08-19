import mujoco
import numpy as np
import copy
import sys
sys.path.append('.')
from testing import doge_utils

class MujocoObject:
    def __init__(self, model, data, bodyname):
        self.model = model
        self.data = data
        self.bodyname = bodyname
        self.bodyid = model.body(bodyname).id
        self.initial_pos = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.set_initial_pos()
    
    def apply_force(self, force, model, data, torque=np.array([0.0, 0.0, 0.0]), point_on_body=np.array([0.0, 0.0, 0.0])):
        mujoco.mj_applyFT(model, data, force, torque, point_on_body, self.bodyid, data.qfrc_applied)
    
    def reset_force(self):
        self.data.qfrc_applied = np.zeros(self.model.nv)
        self.data.qfrc_smooth[self.bodyid : self.bodyid + 6] = np.zeros(6)
        self.data.qvel[self.bodyid : self.bodyid + 6] = np.zeros(6)

    def velocity(self):
        velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.bodyid, velocity, True)
        return velocity
    
    def position(self):
        return self.data.qpos[self.bodyid + 1 : self.bodyid + 8] # [x, y, z, qw, qx, qy, qz]
    
    def set_initial_pos(self):
        print(f'Initial position of {self.bodyname}: {self.position_string(self.position())}')
        self.initial_pos[0:7] = self.position()[0:7]
        self.initial_pos_all = copy.deepcopy(self.data.qpos)

    def set_rotation(self, euler):
        quat = doge_utils.euler_to_quat(euler)
        self.data.qpos[self.bodyid + 3 : self.bodyid + 7] = quat
    
    def reset_position(self):
        self.data.qpos[self.bodyid + 1 : self.bodyid + 8] = self.initial_pos

    def set_position_to(self, position):
        self.data.qpos[self.bodyid + 1 : self.bodyid + 8] = position
        
    def reset_position_all(self):
        self.data.qpos[:] = self.initial_pos_all

    def reset_position_random(self):
        # Randomize y coordinate of object
        self.data.qpos[self.bodyid + 1 : self.bodyid + 8] = self.initial_pos
        self.data.qpos[self.bodyid + 2] = np.random.uniform(-1, 1)

    def get_random_reset_pos(self):
        reset_pos = self.initial_pos
        reset_pos[1] = np.random.uniform(-1, 1)
        return reset_pos

    def position_string(self, pos):
        return f'position of {self.bodyname}: [x: {pos[0]:.2f}, y: {pos[1]:.2f}, z: {pos[2]:.2f}, qw: {pos[3]:.2f}, qx: {pos[4]:.2f}, qy: {pos[5]:.2f}, qz: {pos[6]:.2f}]'
