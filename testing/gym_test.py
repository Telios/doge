import gymnasium as gym
from envs.solo12_v1 import Solo12Env
import torch
from tqdm import tqdm
import numpy as np
import cv2 as cv

env = Solo12Env(xml_file="~/victor/doge/testing/assets/scene.xml",
                render_mode="rgb_array",
                width=128,
                height=128)
obs = env.reset()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in tqdm(range(100)):
    action = np.zeros(12)
    obs, reward, terminated, truncated, info = env.step(action)
    #print(info)
    # scale image
    image = cv.resize(obs["image"], (640, 640))
    image_color = cv.resize(obs["image_color"], (640, 640))
    image_color = cv.cvtColor(image_color, cv.COLOR_RGB2BGR)
    overview_img = cv.resize(obs["overview_img"], (640, 640))
    overview_img = cv.cvtColor(overview_img, cv.COLOR_RGB2BGR)
    concated_img = np.concatenate((image, image_color, overview_img), axis=1)
    cv.imshow("frame", concated_img)
    cv.waitKey(1)
    if terminated:
        env.reset()
env.close()