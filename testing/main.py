import cv2 as cv
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('./testing/assets/scene.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
frames = []
mujoco.mj_resetData(model, data)
bodyid = model.body('projectile1').id
duration = 5
framerate = 30
my_force = np.array([-0.03, 0.0, 0.06])
my_torque = np.array([0.0, 0.0, 0.0])
point_on_body = np.array([0.0, 0.0, 0.0])

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

video_writer = cv.VideoWriter('testing/videos/output.avi', cv.VideoWriter_fourcc(*'XVID'), framerate, (renderer.width, renderer.height))

while data.time < duration:
  if data.time < 0.2:
    mujoco.mj_applyFT(model, data, my_force, my_torque, point_on_body, bodyid, data.qfrc_applied)
  elif data.time > 0.3 and data.time < 0.4:
    data.qfrc_applied[:] = 0.0
  mujoco.mj_step(model, data)
  if len(frames) < data.time * framerate:
    renderer.update_scene(data, scene_option=scene_option)
    pixels = renderer.render()
    # convert to BGR
    pixels2 = cv.cvtColor(pixels, cv.COLOR_RGB2BGR)
    video_writer.write(pixels2)
    frames.append(pixels)

video_writer.release()