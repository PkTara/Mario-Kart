import numpy as np
import cv2
from mss import mss
from PIL import Image
import pygetwindow as gw
import time


from pynput.keyboard import Key, Controller

from random import random
from random import sample

from stable_baselines3 import DQN
import os


class AI_Agent: 
    keymap = {
        "A" : Key.space,
        "B" : "m",
        "left" : ";",
        "right" : "k",
        "do nothing" : "4",
        # "up" : "i",
        # "down" : "k",
        # "start" : Key.enter,
    }  
    moves_list = list(keymap.values())

    print("reading maps...")
    reward_map = cv2.imread('rewards.png', cv2.IMREAD_GRAYSCALE)
    reward_map = cv2.imread('medium_rewards.png', cv2.IMREAD_GRAYSCALE)
    minimap_mask = cv2.imread('minimap_mask.png', cv2.IMREAD_GRAYSCALE)
    reward_folder = "./rewards/"
    reward_counter = 0
    print("finished reading maps")

    no_rewards = len([f for f in os.listdir(reward_folder) if f.endswith('.png')])
    q_table = np.zeros((no_rewards, len(moves_list)))  # stage of the game is the observation, actions are the moves

    learning_rate = 0.1
    discount_factor = 0.95
    exploration_rate = 1.0


    def __init__(self):              
        # self.controller = pyx.vController()
        self.controller = Controller()

    def restart(self):
        self.reward_counter = 0
        self.controller.press(Key.f7)
        time.sleep(0.1)
        self.controller.release(Key.f7)


    def policy(self):
        return self.keymap["do nothing"]
        self.track_distance(screenshot)
        # return self.policy_dqn()
        

    def policy_dqn(self):
        if random() < self.exploration_rate:
            return sample(self.moves_list, 1)[0]
        else:
            state = self.get_state()
            action = self.q_table[state].argmax()
            return self.moves_list[action]
        

    def update_q_table(self, state, action, reward, next_state):
        print(state, action, reward, next_state)

        if self.reward_counter + 1 == self.no_rewards:
            print("Track complete! Resetting...")
            self.restart()
            return

        # Bellman equation
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * self.q_table[next_state].max() - self.q_table[state, action])
        self.reward_counter += 1
        self.exploration_rate *= 0.99  # Decay exploration rate

        



    def get_state(self):
        return self.reward_counter  

    def act(self, screenshot):

        print("Executing policy")
        move = self.policy()
        print(f"Action: {move}")
        # move = sample(self.moves_list, 1)[0]
        self.controller.press(move)
        self.controller.press(Key.space)
        time.sleep(0.3)
        self.controller.release(move)
        self.controller.release(Key.space)

        processed_screenshot = self.process_screenshot(screenshot)




        minimap = self.getMinimap(screenshot)
        reached_goal = self.reachedGoal(minimap)
        print(f"Reached goal: {reached_goal}")
        if reached_goal:
            discrete_move = self.moves_list.index(move)
            self.update_q_table(self.get_state(), discrete_move, 1, self.get_state() + 1)


        
    def track_distance(self, screenshot):
        
        cv2.imshow("Edge Detection", cv2.Canny(screenshot, 50, 150))

    def process_screenshot(self, screenshot):

        gaussian_intensity = (11,11)
        screenshot = cv2.GaussianBlur(screenshot, gaussian_intensity, 0)  
        screenshot = cv2.resize(screenshot, (100,100), interpolation=cv2.INTER_NEAREST )

        edges = cv2.Canny(screenshot, 50, 200)

        cv2.imshow("Small Canny", cv2.resize(edges, (700,700), interpolation=cv2.INTER_NEAREST))
        cv2.moveWindow("Small Canny", 1000, 50)

        return edges


    def process_screenshot_blur(self, screenshot):
        
        gaussian_intensity = (25,25)
        screenshot = cv2.GaussianBlur(screenshot, gaussian_intensity, 0)  
        cv2.imshow("Gaussian Blur", screenshot)

        # screenshot = cv2.bilateralFilter(screenshot, 7, 10, 10)
        # Doesn't work.. Is probably too slow for our usecase, but would be better at preserving edges

        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0] # We only need the hue, to simplify the image further
        # cv2.imshow("Hue", hue)

        screenshot = cv2.resize(hue, (100, 100), interpolation=cv2.INTER_NEAREST)

        screenshot = cv2.resize(screenshot, (700, 700), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Resized Gaussian", screenshot)
        cv2.moveWindow("Resized Gaussian", 1000,250)

        

        return screenshot

    def getMinimap(self, screenshot):
        height, width = screenshot.shape[:2]
        # print(f"Screenshot dimensions: {height}x{width}")
        # print(f"Minimap dimensions: {int(height*0.5)} to {height} {int(width*0.80)} to {int(width*82)}")
     
        minimap_height = [int(height*0.55),int(height*0.808)]
        minimap_width =  [int(width*0.80), int(width*0.88)]
        minimap = screenshot[minimap_height[0]:minimap_height[1], minimap_width[0]:minimap_width[1]]
      
        minimap_mask = cv2.imread('minimap_mask.png', cv2.IMREAD_GRAYSCALE)
        minimap_mask = cv2.resize(minimap_mask, (minimap.shape[1], minimap.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked_minimap = cv2.bitwise_and(minimap, minimap, mask=minimap_mask)
    
        cv2.imshow('Minimap', masked_minimap)

        kart_bbox = self.findKart(masked_minimap)
        if kart_bbox:
            x, y, w, h = kart_bbox
            cv2.rectangle(masked_minimap, (x, y), (x + w, y + h), (0, 255, 0), 2)
      
        cv2.imshow('Red Square Tracking', masked_minimap)

        

        return minimap

    def initController(self):
        pass
    
    def findKart(self, minimap):
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)


        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assuming it's the kart)
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 10:  # threshold to ignore noise
                x, y, w, h = cv2.boundingRect(largest)
                return (x, y, w, h)
        return None
        

    def reachedGoal(self, minimap):
        kart = self.findKart(minimap)
        if not kart:
            return False

        print(f"Reward map: {self.reward_folder + 'reward' +  str(self.reward_counter)}.png")
        self.reward_map = cv2.imread(self.reward_folder + f"reward {self.reward_counter}.png", cv2.IMREAD_GRAYSCALE)
        reward = cv2.resize(self.reward_map, (minimap.shape[1], minimap.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        collision = reward[kart[1]:kart[1]+kart[3], kart[0]:kart[0]+kart[2]]
        # TODO: make kart an obj

        cv2.imshow('Reward', reward)

        

        if np.count_nonzero(collision) > 4:
            print(f"Reached {self.reward_counter} reward!")
            cv2.imshow('Collision', collision)
            return True
        else:
            return False

        # collision = cv2.bitwise_and(minimap, minimap, mask=reward)

        
    


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
    sct_img = np.array(sct.grab(bounding_box))
    if show:
        cv2.imshow('screen', sct_img)            

    return sct_img

if __name__ == "__main__":
    window_title = "gopher64"
    # window_title = "Dinosaur Game"
    record(window_title)