import gym
from gym import spaces
import numpy as np
import pygame
import math
import random

class MoonLanderEnv(gym.Env):
    def __init__(self):
        super(MoonLanderEnv, self).__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 800, 600

        # Lander properties
        self.lander_width, self.lander_height = 20, 40
        self.lander_x = self.WIDTH // 2 - self.lander_width // 2
        self.lander_y = 50
        self.lander_vel_x = 0
        self.lander_vel_y = 0
        self.gravity = 0.2
        self.thrust = 0.3
        self.angle = 0
        self.rotation_speed = 1.5
        self.damping = 0.99

        # Landing pad properties
        self.pad_width, self.pad_height = 100, 20
        self.pad_x = random.randint(0, self.WIDTH - self.pad_width)
        self.pad_y = self.HEIGHT - 50

        # Define action and observation space
        self.action_space = spaces.Discrete(6)  # 0: no action, 1: left, 2: right, 3: weak thrust, 4: medium thrust, 5: strong thrust

        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, -10, -180, 0, 0], dtype=np.float32),  # Explicitly cast to float32
            high=np.array([self.WIDTH, self.HEIGHT, 10, 10, 180, self.WIDTH, self.HEIGHT], dtype=np.float32),  # Explicitly cast to float32
            dtype=np.float32
        )

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Moonlander Game")

        # Rendering control
        self.rendering_enabled = False

    def enable_rendering(self):
        """Enable rendering."""
        self.rendering_enabled = True

    def disable_rendering(self):
        """Disable rendering."""
        self.rendering_enabled = False

    def reset(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Reset lander state
        self.lander_x = self.WIDTH // 2 - self.lander_width // 2
        self.lander_y = 50
        self.lander_vel_x = 0
        self.lander_vel_y = 0
        self.angle = 0

        # Reset landing pad position
        self.pad_x = random.randint(0, self.WIDTH - self.pad_width)
        self.pad_y = self.HEIGHT - 50

        return self._get_obs()


    def step(self, action):
        pygame.event.pump()

        # Apply action
        if action == 1:  # Rotate left
            self.angle += self.rotation_speed
        elif action == 2:  # Rotate right
            self.angle -= self.rotation_speed
        elif action in [3, 4, 5]:  # Apply thrust
            radians = math.radians(self.angle)
            thrust_force = (action - 2) * 0.1  # 0.1 (weak) to 0.3 (strong)
            self.lander_vel_x += thrust_force * math.sin(radians)
            self.lander_vel_y -= thrust_force * math.cos(radians)

        # Apply gravity and damping
        self.lander_vel_y += self.gravity
        self.lander_vel_x *= self.damping
        self.lander_vel_y *= self.damping

        # Update lander position
        self.lander_x += self.lander_vel_x
        self.lander_y += self.lander_vel_y

        obs = self._get_obs()
        reward, done = self._calculate_reward()

        return obs, reward, done, {}


    def _get_obs(self):
        norm_x = self.lander_x / self.WIDTH
        norm_y = self.lander_y / self.HEIGHT
        norm_vx = self.lander_vel_x / 10  # Assume max velocity ~10
        norm_vy = self.lander_vel_y / 10
        norm_angle = self.angle / 180  # Normalize between -1 and 1
        norm_pad_x = self.pad_x / self.WIDTH
        norm_pad_y = self.pad_y / self.HEIGHT

        return np.array([norm_x, norm_y, norm_vx, norm_vy, norm_angle, norm_pad_x, norm_pad_y], dtype=np.float32)


    def _calculate_reward(self):
        # Calculate distance to landing pad center
        pad_center_x = self.pad_x + self.pad_width / 2
        pad_center_y = self.pad_y
        lander_center_x = self.lander_x + self.lander_width / 2
        lander_center_y = self.lander_y + self.lander_height / 2

        distance = math.sqrt((lander_center_x - pad_center_x) ** 2 + (lander_center_y - pad_center_y) ** 2)
        distance_reward = -0.01 * distance  # Small penalty for being far

        # Reward for reducing velocity
        speed_penalty = -0.1 * (abs(self.lander_vel_x) + abs(self.lander_vel_y))

        # Angle penalty (to keep the lander upright)
        angle_penalty = -0.5 * abs(self.angle) if abs(self.angle) > 15 else 0

        # Successful landing reward
        if (self.lander_x + self.lander_width >= self.pad_x and
            self.lander_x <= self.pad_x + self.pad_width and
            self.lander_y + self.lander_height >= self.pad_y):
            if abs(self.lander_vel_x) < 2 and abs(self.lander_vel_y) < 2 and abs(self.angle) < 15:
                return 100 + distance_reward + speed_penalty + angle_penalty, True  # Successful landing
            else:
                return -100, True  # Crash

        # Check for out of bounds
        if (self.lander_y > self.HEIGHT or self.lander_x < 0 or self.lander_x + self.lander_width > self.WIDTH):
            return -100, True  # Out of bounds

        return distance_reward + speed_penalty + angle_penalty, False


    def render(self, mode='human'):
        if not self.rendering_enabled:
            return

        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pad_x, self.pad_y, self.pad_width, self.pad_height))
        self._draw_lander()
        pygame.display.flip()


    def _draw_lander(self):
        # Draw the lander
        tip_x = self.lander_x + self.lander_width // 2
        tip_y = self.lander_y
        left_x = self.lander_x
        left_y = self.lander_y + self.lander_height // 3
        right_x = self.lander_x + self.lander_width
        right_y = self.lander_y + self.lander_height // 3
        bottom_left_x = self.lander_x
        bottom_left_y = self.lander_y + self.lander_height
        bottom_right_x = self.lander_x + self.lander_width
        bottom_right_y = self.lander_y + self.lander_height

        rocket_points = [
            (tip_x, tip_y),
            (left_x, left_y),
            (bottom_left_x, bottom_left_y),
            (bottom_right_x, bottom_right_y),
            (right_x, right_y),
        ]

        center = (self.lander_x + self.lander_width // 2, self.lander_y + self.lander_height // 2)
        rotated_points = []
        for point in rocket_points:
            translated_x = point[0] - center[0]
            translated_y = point[1] - center[1]
            radians = math.radians(self.angle)
            rotated_x = translated_x * math.cos(radians) - translated_y * math.sin(radians)
            rotated_y = translated_x * math.sin(radians) + translated_y * math.cos(radians)
            rotated_x += center[0]
            rotated_y += center[1]
            rotated_points.append((rotated_x, rotated_y))

        pygame.draw.polygon(self.screen, (255, 255, 255), rotated_points)

    def close(self):
        pygame.quit()