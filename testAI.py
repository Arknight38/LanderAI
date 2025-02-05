from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from moonlander_env import MoonLanderEnv

# Load the trained model
model = PPO.load("moonlander_ai")

# Create the test environment with rendering enabled
def make_test_env():
    return MoonLanderEnv(render_mode="human")  # Enable rendering

env = DummyVecEnv([make_test_env])  # Wrap for compatibility

# If training used VecNormalize, load normalization parameters
env = VecNormalize.load("moonlander_ai_norm.pkl", env)
env.training = False  # Disable training mode (keeps rewards true)
env.norm_reward = False  # Use raw rewards during testing

# Test the model for a fixed number of episodes
num_episodes = 5
for episode in range(num_episodes):
    obs = env.reset()  # No need to unpack with an underscore here
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Deterministic for consistency
        obs, rewards, dones, infos = env.step(action)  # Correctly unpacking four values
        done = dones[0]  # Extract first environment's done flag

# Close the environment
env.close()
