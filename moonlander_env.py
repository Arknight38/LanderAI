import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random


class MoonLanderEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window = None

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

        # Action and observation space
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, -10, -180, 0, 0], dtype=np.float32),
            high=np.array([self.WIDTH, self.HEIGHT, 10, 10, 180, self.WIDTH, self.HEIGHT], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize pygame only if rendering is enabled
        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Moonlander Game")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.lander_x = self.WIDTH // 2 - self.lander_width // 2
        self.lander_y = 50
        self.lander_vel_x = 0
        self.lander_vel_y = 0
        self.angle = 0

        self.pad_x = random.randint(0, self.WIDTH - self.pad_width)
        self.pad_y = self.HEIGHT - 50

        # Initialize pygame display only if rendering is enabled
        if self.render_mode == "human" and not hasattr(self, "window"):
            pygame.init()
            self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        if self.render_mode == "human":
            pygame.event.pump()  # Process events only if rendering

        if action == 1:
            self.angle += self.rotation_speed
        elif action == 2:
            self.angle -= self.rotation_speed
        elif action in [3, 4, 5]:
            radians = math.radians(self.angle)
            thrust_force = (action - 2) * 0.1
            self.lander_vel_x += thrust_force * math.sin(radians)
            self.lander_vel_y -= thrust_force * math.cos(radians)

        self.lander_vel_y += self.gravity
        self.lander_vel_x *= self.damping
        self.lander_vel_y *= self.damping

        self.lander_x += self.lander_vel_x
        self.lander_y += self.lander_vel_y

        obs = self._get_obs()
        reward, terminated = self._calculate_reward()

        truncated = self.lander_y > self.HEIGHT or self.lander_x < 0 or self.lander_x + \
            self.lander_width > self.WIDTH

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        norm_x = self.lander_x / self.WIDTH
        norm_y = self.lander_y / self.HEIGHT
        norm_vx = self.lander_vel_x / 10
        norm_vy = self.lander_vel_y / 10
        norm_angle = self.angle / 180
        norm_pad_x = self.pad_x / self.WIDTH
        norm_pad_y = self.pad_y / self.HEIGHT

        return np.array([norm_x, norm_y, norm_vx, norm_vy, norm_angle, norm_pad_x, norm_pad_y], dtype=np.float32)

    def _calculate_reward(self):
        pad_center_x = self.pad_x + self.pad_width / 2
        pad_center_y = self.pad_y
        lander_center_x = self.lander_x + self.lander_width / 2
        lander_center_y = self.lander_y + self.lander_height / 2

        distance = math.sqrt((lander_center_x - pad_center_x)
                             ** 2 + (lander_center_y - pad_center_y) ** 2)
        distance_reward = -0.01 * distance

        speed_penalty = -0.1 * \
            (abs(self.lander_vel_x) + abs(self.lander_vel_y))
        angle_penalty = -0.5 * abs(self.angle) if abs(self.angle) > 15 else 0

        if (self.lander_x + self.lander_width >= self.pad_x and
            self.lander_x <= self.pad_x + self.pad_width and
                self.lander_y + self.lander_height >= self.pad_y):
            if abs(self.lander_vel_x) < 2 and abs(self.lander_vel_y) < 2 and abs(self.angle) < 15:
                return 100 + distance_reward + speed_penalty + angle_penalty, True
            else:
                return -100, True

        return distance_reward + speed_penalty + angle_penalty, False

    def render(self):
        if self.render_mode != "human":
            return

        self.window.fill((0, 0, 0))
        pygame.draw.rect(self.window, (0, 255, 0), (self.pad_x,
        self.pad_y, self.pad_width, self.pad_height))
        self._draw_lander()
        pygame.display.flip()

    def _draw_lander(self):
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

        center = (self.lander_x + self.lander_width // 2,
                  self.lander_y + self.lander_height // 2)
        rotated_points = []
        for point in rocket_points:
            translated_x = point[0] - center[0]
            translated_y = point[1] - center[1]
            radians = math.radians(self.angle)
            rotated_x = translated_x * \
                math.cos(radians) - translated_y * math.sin(radians)
            rotated_y = translated_x * \
                math.sin(radians) + translated_y * math.cos(radians)
            rotated_x += center[0]
            rotated_y += center[1]
            rotated_points.append((rotated_x, rotated_y))

        pygame.draw.polygon(self.window, (255, 255, 255), rotated_points)

    def close(self):
        if self.window:
            pygame.quit()
