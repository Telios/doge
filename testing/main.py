import cv2 as cv
import mujoco
import numpy as np
from mujoco_object import MujocoObject
import utils
import torch, time, esim_torch

model = mujoco.MjModel.from_xml_path('./testing/assets/scene.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, width=640, height=480)
renderer.update_scene(data, 0)
frames = []
mujoco.mj_resetData(model, data)
mujoco.mj_step(model, data)


projectile = MujocoObject(model, data, 'projectile1')
esim = esim_torch.ESIM(
  0.2,
  0.2,
  0,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
duration = 200
framerate = 30
save_video = False
my_force = np.array([-0.4, 0.0, 1.1])

scene_option = mujoco.MjvOption()

if save_video:
  video_writer = cv.VideoWriter('testing/videos/output.avi', cv.VideoWriter_fourcc(*'XVID'), framerate, (renderer.width, renderer.height))
resetted = False

def get_event_image(img_in, timestamp):
  img = cv.cvtColor(img_in, cv.COLOR_RGB2GRAY)
  img = np.log(img.astype("float32") / 255 + 1e-4)
  img = torch.from_numpy(img).to(DEVICE)
  timestamp = torch.tensor([timestamp], device=DEVICE, dtype=torch.int64)

  events = esim.forward(img, torch.tensor([timestamp], device=DEVICE))
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

starttime = time.time_ns()
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
  timestamp_ns = time.time_ns() - starttime
  if len(frames) < data.time * framerate:
    renderer.update_scene(data, scene_option=scene_option, camera='front-camera')
    front_camera = renderer.render()
    event_image = get_event_image(front_camera, timestamp_ns)
    renderer.update_scene(data, scene_option=scene_option)
    overview_cam = renderer.render()
    overview_cam = cv.cvtColor(overview_cam, cv.COLOR_RGB2BGR)
    front_camera = cv.cvtColor(front_camera, cv.COLOR_RGB2BGR)
    concat_img = np.concatenate((overview_cam, front_camera, event_image), axis=1)
    if save_video:
      video_writer.write(concat_img)
    cv.imshow('frame', concat_img)
    cv.waitKey(1)
    frames.append(concat_img)

if save_video:
  video_writer.release()