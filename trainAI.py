from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
from moonlander_env import MoonLanderEnv
import gymnasium as gym

# Helper function to create an environment


def make_env():
    def _init():
        env = MoonLanderEnv(render_mode=None)  # Disable rendering for training
        return gym.wrappers.RecordEpisodeStatistics(env)  # Track stats
    return _init


if __name__ == "__main__":
    # Use DummyVecEnv for training
    env = DummyVecEnv([make_env() for _ in range(4)])

    # Normalize observations and rewards for stable learning
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Initialize PPO model with GPU support
    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log="./ppo_moonlander/", device="auto")

    # Create a separate evaluation environment
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True,
                            norm_reward=True, clip_obs=10.0)

    # Add EvalCallback to evaluate the model every 10,000 steps
    early_stopping_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=10, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        callback_on_new_best=early_stopping_callback  # Attach early stopping here
    )

    callback = CallbackList([eval_callback])

    # Training loop with checkpoints
    TIMESTEPS = 10000
    for i in range(10):
        model.learn(total_timesteps=TIMESTEPS,
                    callback=callback, reset_num_timesteps=False)
        model.save(f"moonlander_ai_{(i+1) * TIMESTEPS}")

    # Save final model and normalization parameters
    model.save("moonlander_ai")
    env.save("moonlander_ai_norm.pkl")

    # --- TESTING PHASE ---
    # Create a new environment with rendering enabled
    test_env = DummyVecEnv([lambda: MoonLanderEnv(render_mode="human")])

    # Load normalization parameters (if used during training)
    test_env = VecNormalize.load("moonlander_ai_norm.pkl", test_env)
    test_env.training = False  # Disable training mode for normalization
    test_env.norm_reward = False  # Disable reward normalization for testing

    # Load the trained model
    model = PPO.load("moonlander_ai", env=test_env)

    # Test the model
    # Test the model
    num_episodes = 5
    for episode in range(num_episodes):
        obs = test_env.reset()  # Remove tuple unpacking
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = test_env.step(
                action)  # Fix step unpacking
            done = dones[0]  # Track only first environment
            test_env.render()  # Render the environment

    test_env.close()
