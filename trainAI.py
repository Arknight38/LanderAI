from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from moonlander_env import MoonLanderEnv

def make_env():
    return MoonLanderEnv()

if __name__ == "__main__":
    # Use multiple environments for better generalization
    env = SubprocVecEnv([make_env for _ in range(4)])  
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)  # Normalize for stability

    # Initialize PPO model with GPU support
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_moonlander/", device="auto")

    # Save checkpoints during training
    TIMESTEPS = 10000
    for i in range(10):  
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"moonlander_ai_{(i+1) * TIMESTEPS}")

    # Save final model
    model.save("moonlander_ai")

    # Test trained model
    env.training = False  # Disable normalization during testing
    env.norm_reward = False
    env.get_attr("enable_rendering")[0]()  # Calls enable_rendering() on the first environment


    num_episodes = 5  # Test for 5 landings
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            env.render()

    # Close environment
    env.close()
