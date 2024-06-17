import cv2 as cv
import mujoco
import numpy as np
from mujoco_object import MujocoObject
import utils

model = mujoco.MjModel.from_xml_path('./testing/assets/scene.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, width=640, height=480)
renderer.update_scene(data, 0)
frames = []
mujoco.mj_resetData(model, data)
mujoco.mj_step(model, data)


projectile = MujocoObject(model, data, 'projectile1')

duration = 200
framerate = 10
save_video = False
my_force = np.array([-0.4, 0.0, 1.1])

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

if save_video:
  video_writer = cv.VideoWriter('testing/videos/output.avi', cv.VideoWriter_fourcc(*'XVID'), framerate, (renderer.width, renderer.height))
resetted = False

while data.time < duration:
  if data.time % 5 < 0.2:
    resetted = False
    projectile.apply_force(my_force)
  elif data.time % 5 > 0.3 and data.time % 5 < 0.4:
    projectile.reset_force()
  elif data.time % 5 > 4.0 and not resetted:
    projectile.reset_position_random()
    my_force = utils.calculate_force_vector(data.body('base_link').xpos, projectile.position())
    resetted = True
  mujoco.mj_step(model, data)
  if len(frames) < data.time * framerate:
    renderer.update_scene(data, scene_option=scene_option, camera='front-camera')
    pixels = renderer.render()
    renderer.update_scene(data, scene_option=scene_option)
    pixels = np.concatenate((pixels, renderer.render()), axis=1)
    # convert to BGR
    pixels2 = cv.cvtColor(pixels, cv.COLOR_RGB2BGR)
    if save_video:
      video_writer.write(pixels2)
    cv.imshow('frame', pixels2)
    cv.waitKey(1)
    frames.append(pixels)

if save_video:
  video_writer.release()