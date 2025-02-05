from stable_baselines3 import PPO
from moonlander_env import MoonLanderEnv

# Create the environment
env = MoonLanderEnv()

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("moonlander_ai")

# Close the environment
env.close()