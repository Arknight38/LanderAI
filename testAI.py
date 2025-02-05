from stable_baselines3 import PPO
from moonlander_env import MoonLanderEnv

# Load the trained model
model = PPO.load("moonlander_ai")

# Create the environment
env = MoonLanderEnv()

# Test the model
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Close the environment
env.close()