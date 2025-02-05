from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from moonlander_env import MoonLanderEnv

# Load the trained model
model = PPO.load("moonlander_ai")

# Create the environment
env = DummyVecEnv([lambda: MoonLanderEnv()])  # Wrap for compatibility

# If training used VecNormalize, load normalization parameters
env = VecNormalize.load("moonlander_ai_norm.pkl", env)
env.training = False  # Disable training mode (keeps rewards true)
env.norm_reward = False  # Use raw rewards during testing

# Enable rendering on the first environment inside the VecEnv
env.get_attr("enable_rendering")[0]()  # Correct way to call enable_rendering()

# Test the model for a fixed number of episodes
num_episodes = 5
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Deterministic for consistency
        obs, _, done, _ = env.step(action)
        env.render()  # Use render() directly (No need for get_attr here)

# Close the environment
env.close()
