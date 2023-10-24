from gym_coursedesign.environment import MultiAgentEnv
import gym_coursedesign.scenarios as scenarios
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
class MyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,render_mode="rgb_array"):
        scenario = scenarios.load('simple_tag.py').Scenario()
        # create world
        world = scenario.make_world()
        # create gym_coursedesign environment
        self.multienv = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                            done_callback=scenario.is_done)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=0, high=255, shape=(300, 300,1), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def step(self,action):
        obs_n, reward_n, done_n, _ = self.multienv.step(action)
        img = np.array(self.multienv.render("rgb_array")).reshape(800,800,3)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(img_gray, (300, 300)).reshape(300, 300,1)
        return obs,reward_n[1],any(done_n),False,_

    def reset(self,seed=None, options=None):
        self.multienv.reset()
        img = np.array(self.multienv.render("rgb_array")).reshape(800,800,3)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(img_gray, (300, 300)).reshape(300, 300,1)
        _ = {}
        return obs,_

    def render(self):
        if self.render_mode == "rgb_array":
            return np.array(self.multienv.render("rgb_array")).reshape(800,800,3)

    def _get_obs(self):
        img = np.array(self.multienv.render("rgb_array")).reshape(800,800,3)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(img_gray, (300, 300)).reshape(300, 300,1)
        return obs


if __name__ == "__main__":
    env = MyEnv()
    obs = env.reset()
    import time
    while True:
        obs, reward, done, _ ,_= env.step(env.action_space.sample())
        print(obs)
        # time.sleep(0.05)
        if done:
            env.reset()

