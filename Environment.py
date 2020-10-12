from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

import json
import signal
import atexit

import cv2
import base64
import io

import numpy as np

import math

def base642numpy(obs):
    im_bytes = base64.b64decode(obs)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (80, 50))[14:-4,8:-8]
    img = img[np.newaxis, np.newaxis, ...]

    return img.astype(float) / 255

class Environment:
    def __init__(self):
        self.driver = webdriver.Firefox(executable_path = './firefox-driver/geckodriver')

        signal.signal(signal.SIGINT, self.quit)
        signal.signal(signal.SIGTERM, self.quit)
        signal.signal(signal.SIGTSTP, self.quit)
        atexit.register(self.quit)

        self.driver.get('http://localhost:3000?webgl=true')

        self.score_div = self.driver.find_element_by_id('score')
        self.terminal_div = self.driver.find_element_by_id('terminal')

        try:
            # WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.ID, 'window1')))
            condition = EC.text_to_be_present_in_element((By.ID, "terminal"), "FALSE")
            WebDriverWait(self.driver, 10).until(condition)
            print("Page is ready")
        except TimeoutException:
            print("Loading took too much time")

        self.canvas = self.driver.find_element_by_id('window1')

        # self.position = float(self.score_div.get_attribute("innerHTML"))
        self.velocity = 0

        self.last_obs = None

    def quit(self):
        self.driver.quit()

    def get_reward(self):
        new_pos = float(self.score_div.get_attribute("innerHTML"))
        new_vel = (new_pos - self.position)
        accel = new_vel - self.velocity

        self.position = new_pos
        self.velocity = new_vel

        reward = new_vel

        return reward, new_pos, new_vel, accel

    def step(self, action):
        action = action[0]
        new_position = float(self.score_div.get_attribute("innerHTML"))
        new_vel = (new_position - self.position)
        accel = new_vel - self.velocity

        reward, _, _, _ = self.get_reward()

        action_chain = ActionChains(self.driver) 

        if action[0] == 1: action_chain.key_down('q')
        else: action_chain.key_up('q')

        if action[1] == 1: action_chain.key_down('w')
        else: action_chain.key_up('w')

        if action[2] == 1: action_chain.key_down('o')
        else: action_chain.key_up('o')

        if action[3] == 1: action_chain.key_down('p')
        else: action_chain.key_up('p')

        action_chain.perform()

        return self.obs(), reward, self.terminal_div.get_attribute("innerHTML").lower() == "true", new_position

    def obs(self):
        new_obs = base642numpy(self.canvas.screenshot_as_base64)

        if self.last_obs is None:
            self.last_obs = np.tile(new_obs, (1,3,1,1))
        else:
            self.last_obs = np.concatenate([self.last_obs[:,1:,:,:], new_obs], axis=1)

        return self.last_obs

    def reset(self):
        self.canvas.click()
        ActionChains(self.driver).key_up('p').key_up('o').key_up('w').key_up('q').key_down('r').key_up('r').perform()

        self.position = float(self.score_div.get_attribute("innerHTML"))
        self.velocity = 0
        self.last_obs = None

        return self.obs()
