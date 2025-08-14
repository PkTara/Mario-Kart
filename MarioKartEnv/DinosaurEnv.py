from typing import Optional
import numpy as np
import gymnasium as gym
from pynput import keys

import mss 
import time
import getpythonwindows as gw



class DinosaurEnv(gym.Env):
    
     def __init__(self, window):

         self.observation_space = gym.spaces.Dict({
              "image" : gym.spaces.Box(low=0, high=255, shape= (window.height, window.width, channels:=3), dtype=np.uint8) 
         })

         self.action_space = gym.spaces.Discrete(9)
         self._action_to_direction = {
                0: keys.space,
                1: "m",
                2: keys.enter,
                3: "q",
                4: "e",
                5: "w",
                6: "a",
                7: "s",
                8: "d"
                }
         
         self.sct = mss()

         self.window = self.initialse_window(window.title, sct)


         
     def initialse_window(window_title, sct):
    
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
        image = take_screenshot(self.sct, bounding_box)
        image = image[:,:, :3] # remove alpha channel

        return { "image" : image }
     
     def reset():
         input("Please reset the environment")
         pass # hah we don't do that here
     
     def step(self, action):
         self.controller.press(self._action_to_direction[action])
         
         terminated = False
         reward = 0
         observation = self._get_obs()
         info = False

         return observation, reward, terminated, info