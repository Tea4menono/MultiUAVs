import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

user_positions = [
    [[64, 12], [56, 22], [44, 18], [49, 27], [52, 49], [55, 50], [48, 40], [45, 61], [44, 41], [48, 57], [30, 56],
     [51, 29], [59, 63], [54, 67], [32, 55], [56, 53], [57, 22], [53, 8], [63, 45], [54, 52]],
    [[4, 19], [38, 14], [22, 45], [23, 33], [1, 46], [38, 19], [4, 3], [20, 57], [23, 27], [30, 7], [22, 31],
     [5, 55], [33, 25], [41, 3], [10, 20], [32, 19], [5, 18], [5, 21], [5, 24], [31, 26], [0, 33], [15, 20],
     [0, 60], [45, 4], [2, 19], [24, 31], [25, 18], [9, 12], [7, 4], [0, 14]]]
uav_positions = [
    [49.86953366, 32.88788339],
    [14.11916185, 25.57281582]
]


class StepsEnv(gymnasium.Env):
    def __init__(self, user_number):
        super(StepsEnv, self).__init__()
        self.user_number = user_number
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.user_number,), dtype=np.float32)
        self.observation_space = spaces.Dict({"power": spaces.Box(low=0, high=10, shape=(self.user_number,), dtype=np.float32)})
        self.state = {"power": np.zeros(self.user_number)}

    def reset(self, **kwargs):
        self.state = {"power": np.zeros(self.user_number)}
        return self.state, {}

    def step(self, action):
        self.state["power"] += action
        self.state["power"] = np.clip(self.state["power"], 0, 10)

        # reward = served_user_number

        reward = 1
        print(self.state)

        return self.state, reward, False, False, {}

    def close(self):
        pass


# Create the custom environment
env = StepsEnv()

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)

# Train the model with DDPG
model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)

# Save the model
model.save("ddpg_steps")

# Load the model
del model  # remove to demonstrate saving and loading
model = DDPG.load("ddpg_steps")

# Run the trained model
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    if done:
        print("Goal Reached!")
        break
