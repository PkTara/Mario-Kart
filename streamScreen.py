import numpy as np
import cv2
from mss import mss
from PIL import Image
import pygetwindow as gw
import time
import pyxinput as pyx

from pynput.keyboard import Key, Controller

from random import randint


class AI_Agent: 
    keymap = {
        "A" : Key.space,
        "B" : "m",
        "left" : "j",
        "right" : "l",
        "up" : "i",
        "down" : "k",
        "start" : "enter",
    }                 
    def __init__(self):              
        # self.controller = pyx.vController()
        self.controller = Controller()

    def act(self, screenshot):
        if randint(0, 100) < 50:
            self.controller.press(Key.space)
            time.sleep(0.1)
            self.controller.release(Key.space)

    def initController(self):
        pass
        


ai_agent = AI_Agent()
ai_agent.initController()

def record(window_title):
    sct = mss()

    window = gw.getWindowsWithTitle(window_title)[0]
    
    if window.isMinimized:
        window.restore()
        time.sleep(0.5) 

    window.activate()
    time.sleep(0.5)



    bounding_box = {
                    'top': window.top,
                    'left': window.left,
                    'width': window.width,
                    'height': window.height
                }


    while True:
        fps = 30
        time.sleep(1/fps)

        screenshot = take_screenshot(sct, bounding_box)
        if ai_agent:
            ai_agent.act(screenshot)


        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            return False
        

def take_screenshot(sct, bounding_box, show=True):
    sct_img = sct.grab(bounding_box)
    if show:
        cv2.imshow('screen', np.array(sct_img))            

    return sct_img

if __name__ == "__main__":
    window_title = "gopher64"
    # window_title = "Dinosaur Game"
    record(window_title)