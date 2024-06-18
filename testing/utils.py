import numpy as np
from scipy.spatial.transform import Rotation

def euler_to_quat(euler):
    """Convert Euler angles (rad) to quaternion."""
    r = Rotation.from_euler('xyz', euler)
    return r.as_quat()

def calculate_force_vector(goal_pos, current_pos, force_x=-0.7, force_z=0.9):
    """Calculate force vector to reach goal position."""
    force = np.array([0.0, 0.0, 0.0])
    force[0] = force_x
    force[1] = -current_pos[1] * 0.2
    force[2] = force_z
    return force
