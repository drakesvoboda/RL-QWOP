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


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

def base642numpy(obs):
    im_bytes = base64.b64decode(obs)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (80, 50))[14:-4,8:-8]
    img = img[np.newaxis, np.newaxis, ...]

    return img.astype(float) / 255

class QWOPEnvironment:
    def __init__(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, './firefox-driver/geckodriver')
        self.driver = webdriver.Firefox(executable_path=filename)

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)
        signal.signal(signal.SIGTSTP, self.close)
        atexit.register(self.close)

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

    @property
    def obs_dim(self):
        return (3, 32, 64)

    @property   
    def act_dim(self):
        return 4

    def close(self):
        self.driver.quit()

    def get_state(self):
        new_pos = float(self.score_div.get_attribute("innerHTML"))
        new_vel = (new_pos - self.position)
        accel = new_vel - self.velocity

        self.position = new_pos
        self.velocity = new_vel

        terminal = self.terminal_div.get_attribute("innerHTML").lower() == "true"

        return self.position, self.velocity, accel, terminal

    def step(self, action):
        action = action[0]

        obs = self.obs()
        pos, vel, accel, terminal = self.get_state()

        reward = -0.5 if terminal else (10 * max(0, vel))**2

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

        return obs, reward, terminal, pos

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

        time.sleep(0.01)

        self.position = float(self.score_div.get_attribute("innerHTML"))
        self.velocity = 0
        self.last_obs = None

        return self.obs()
