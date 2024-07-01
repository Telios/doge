import numpy as np
from scipy.spatial.transform import Rotation

def euler_to_quat(euler):
    """Convert Euler angles (rad) to quaternion."""
    r = Rotation.from_euler('xyz', euler)
    return r.as_quat()

def quat_to_euler(quat):
    """Convert quaternion to Euler angles (rad)."""
    r = Rotation.from_quat(quat)
    return r.as_euler('xyz')

def calculate_force_vector(goal_pos, current_pos, force_x=-3.7, force_z=3.9):
    """Calculate force vector to reach goal position."""
    force = np.array([0.0, 0.0, 0.0])
    y_diff = goal_pos[1] - current_pos[1]
    force[0] = force_x
    force[1] = y_diff
    force[2] = force_z
    return force