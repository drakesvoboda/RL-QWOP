import os
import io
import math
import cv2
import base64
import json
import signal
import atexit
import time
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException


def base642numpy(obs, size=(36, 36), flags=cv2.IMREAD_COLOR):
    im_bytes = base64.b64decode(obs)
    # im_arr is one-dim Numpy array
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=flags)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    img = img[90:-22, 176:-176, :]
    img = cv2.resize(img, size)

    return img.astype(np.uint8)


class QWOPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        profile = webdriver.FirefoxProfile()
        profile.set_preference("media.volume_scale", "0.0")
        self.driver = webdriver.Firefox(firefox_profile=profile, executable_path=os.path.join(os.path.dirname(__file__), './firefox-driver/geckodriver'))

        self.action_space = spaces.MultiBinary(4)

        high = np.array([np.inf]*9)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32) 

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)
        signal.signal(signal.SIGTSTP, self.close)
        atexit.register(self.close)

        self.driver.get('http://localhost:3000?webgl=true')

        try:
            # WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.ID, 'window1')))
            condition = EC.text_to_be_present_in_element(
                (By.ID, "terminal"), "false")
            WebDriverWait(self.driver, 10).until(condition)
            print("Page is ready")
        except TimeoutException:
            print("Loading took too much time")

        self.canvas =       self.driver.find_element_by_id('window1')
        self.terminal_div = self.driver.find_element_by_id("terminal")
        self.state =        self.driver.find_element_by_id("state")

        self.max_fps = 8
        self.last_call = time.time()
        self.last_state = time.time()
        self.old_posx = 0
        self.old_velx = 0

        self.get_state()

    def get_state(self):
        state = json.loads(self.state.get_attribute("innerHTML"))
        state["terminal"] = self.terminal_div.get_attribute("innerHTML").lower() == "true"
        
        current_time = time.time()
        delta = current_time - self.last_state
        self.last_state = current_time
        
        state["diffx"] = state["posx"] - self.old_posx

        state["velx"] = (state["diffx"]) / delta
        state["accelx"] = (state["velx"] - self.old_velx) / delta

        self.old_posx = state["posx"]
        self.old_velx = state["velx"]

        return state

    def compute_reward(self, state=None):
        state = state if state != None else self.get_state()
        reward = (3*state["diffx"])**2 if state["diffx"] > 0 else 0

        # angle = state["torso"] + (math.pi / 2)
        # if abs(angle) > (math.pi / 2): reward -= abs(angle)

        reward = -0.5 if state["terminal"] else reward

        return reward

    def throttle(self):
        call_time = time.time()
        delta = call_time - self.last_call    
        min_delta = 1 / self.max_fps
        if delta < min_delta: time.sleep(min_delta - delta)
        self.last_call = call_time

        return True

    def step(self, action):
        self.throttle()

        state = self.get_state()
        obs = self.obs(state)
        reward = self.compute_reward(state)
        action_chain = ActionChains(self.driver)

        if action[0] == 1:
            action_chain.key_down('q')
        else:
            action_chain.key_up('q')

        if action[1] == 1:
            action_chain.key_down('w')
        else:
            action_chain.key_up('w')

        if action[2] == 1:
            action_chain.key_down('o')
        else:
            action_chain.key_up('o')

        if action[3] == 1:
            action_chain.key_down('p')
        else:
            action_chain.key_up('p')

        action_chain.perform()

        return obs, reward, state["terminal"], { "state": state }

    def obs(self, state=None):
        state = state if state != None else self.get_state()

        return np.array([state["posy"], 
                        state["head"], 
                        state["torso"], 
                        state["leftarm"],              
                        state["rightarm"], 
                        state["leftthigh"], 
                        state["rightthigh"], 
                        state["leftcalf"], 
                        state["rightcalf"]], 
                        dtype=np.float32)

    def reset(self):
        self.canvas.click()

        ActionChains(self.driver).key_up('p').key_up('o').key_up('w').key_up('q').key_down('r').key_up('r').perform()

        time.sleep(0.03)

        state = self.get_state()

        self.last_call = time.time()
        self.old_velx = 0

        return self.obs()

    def render(self, mode='human'): pass

    def close(self, *args, **kwargs):
        self.driver.quit()

class FrameQWOPEnv(QWOPEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(36, 36, 3), dtype=np.uint8)

    def obs(self, state=None):
        return base642numpy(self.canvas.screenshot_as_base64)

class MultiFrameQWOPEnv(QWOPEnv):
    def __init__(self):
        super().__init__()
        self.last_obs = None
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(36, 36, 3), dtype=np.uint8)

    def obs(self, state=None):
        new_obs = base642numpy(
            self.canvas.screenshot_as_base64, flags=cv2.IMREAD_GRAYSCALE)
        new_obs = new_obs[:, :, np.newaxis]

        if self.last_obs is None:
            self.last_obs = np.tile(new_obs, (1, 1, 3))
        else:
            self.last_obs = np.concatenate(
                [self.last_obs[:, :, 1:], new_obs], axis=2)

        return self.last_obs

    def reset(self):
        self.last_obs = None
        return super().reset()
