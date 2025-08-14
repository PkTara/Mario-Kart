from typing import Optional
import numpy as np
import gymnasium as gym
from pynput.keyboard import Key

import mss 
import time
import pygetwindow as gw
import cv2

class MarioKartEnv(gym.Env):
    
     def __init__(self, window):

         self.action_space = gym.spaces.Discrete(9)
         self._action_to_direction = {
                0: Key.space,
                1: "m",
                2: Key.enter,
                3: "q",
                4: "e",
                5: "w",
                6: "a",
                7: "s",
                8: "d"
                }
         
         self.sct = mss()

         self.window = self.initialise_window(window.title, sct)

         self.observation_space = gym.spaces.Dict({
              "image" : gym.spaces.Box(low=0, high=255, shape= (window.height, window.width, channels:=3), dtype=np.uint8) 
         })
         
     def initialise_window(window_title, sct):
    
        window = gw.getWindowsWithTitle(window_title)[0]
        
        if window.isMinimized:
            window.restore()
            time.sleep(0.5) 

        window.activate()
        time.sleep(0.5)

        return window
         



     def _get_obs(self):
        bounding_box = {
            'top': self.window.top,
            'left': self.window.left,
            'width': self.window.width,
            'height': self.window.height
        }

        image = self.take_screenshot(self.sct, bounding_box, show=self.render_mode=="human")
        image = image[:,:, :3] # remove alpha channel

        return { "image" : image }
     
     def take_screenshot(sct, bounding_box, show=True):
        sct_img = sct.grab(bounding_box)
        if show:
            cv2.imshow('screen', np.array(sct_img))            

        return sct_img
     


     def reset(self):
         # Wow, isn't this fugly
         self.controller.press("alt")
         self.controller.press("2")
         self.controller.release("alt")
         self.controller.release("2")

         self.controller.press("f7")
         self.controller.release("f7")


     
     def step(self, action):
         self.controller.press(self._action_to_direction[action])
         
         terminated = False
         reward = 0
         observation = self._get_obs()
         info = False

         return observation, reward, terminated, info
     
         def close(self):
             cv2.destroyAllWindows()


