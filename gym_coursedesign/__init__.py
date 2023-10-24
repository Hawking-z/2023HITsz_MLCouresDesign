from gym_coursedesign.envs.coursedesign_env import MyEnv
from gymnasium.envs.registration import register

# Multiagent envs
# ----------------------------------------

register(
    id="gym_coursedesign/CourseDesign_Env-v0",
    entry_point="gym_coursedesign.envs:MyEnv",
    max_episode_steps=200,
)