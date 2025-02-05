import pygame
import sys
import math
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Moonlander Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Font for debug display
font = pygame.font.SysFont("Arial", 14)  # Smaller font size

# Lander properties
lander_width, lander_height = 20, 40  # Narrower width for a rocket shape
lander_x = WIDTH // 2 - lander_width // 2
lander_y = 50
lander_vel_x = 0
lander_vel_y = 0
gravity = 0.2  # Increased gravity for heavier feel
thrust = 0.3  # Reduced thrust for heavier feel
angle = 0  # Initial angle (in degrees)
rotation_speed = 1.5  # Slightly reduced rotation speed
damping = 0.99  # Velocity damping to simulate air resistance

# Landing pad properties
pad_width, pad_height = 100, 20
pad_x = random.randint(0, WIDTH - pad_width)
pad_y = HEIGHT - 50

# Function to draw the rocket-shaped lander
def draw_lander(surface, x, y, angle):
    # Define the points for the rocket shape
    tip_x = x + lander_width // 2
    tip_y = y
    left_x = x
    left_y = y + lander_height // 3
    right_x = x + lander_width
    right_y = y + lander_height // 3
    bottom_left_x = x
    bottom_left_y = y + lander_height
    bottom_right_x = x + lander_width
    bottom_right_y = y + lander_height

    # Create a list of points for the rocket shape
    rocket_points = [
        (tip_x, tip_y),
        (left_x, left_y),
        (bottom_left_x, bottom_left_y),
        (bottom_right_x, bottom_right_y),
        (right_x, right_y),
    ]

    # Rotate the points around the center of the lander
    center = (x + lander_width // 2, y + lander_height // 2)
    rotated_points = []
    for point in rocket_points:
        # Translate point to origin
        translated_x = point[0] - center[0]
        translated_y = point[1] - center[1]
        # Rotate point
        radians = math.radians(angle)
        rotated_x = translated_x * math.cos(radians) - translated_y * math.sin(radians)
        rotated_y = translated_x * math.sin(radians) + translated_y * math.cos(radians)
        # Translate point back
        rotated_x += center[0]
        rotated_y += center[1]
        rotated_points.append((rotated_x, rotated_y))

    # Draw the rocket shape
    pygame.draw.polygon(surface, WHITE, rotated_points)

# Function to draw debug information
def draw_debug_info(surface, x, y, vel_x, vel_y, angle, gravity, thrust_active):
    debug_text = [
        f"X: {x:.2f}",
        f"Y: {y:.2f}",
        f"VX: {vel_x:.2f}",
        f"VY: {vel_y:.2f}",
        f"Angle: {angle:.2f}°",
        f"Gravity: {gravity:.2f}",
        f"Thrust: {'ON' if thrust_active else 'OFF'}",
    ]
    # Render each line of text
    for i, text in enumerate(debug_text):
        text_surface = font.render(text, True, WHITE)
        # Position debug menu in the bottom-left corner
        surface.blit(text_surface, (10, HEIGHT - 150 + i * 20))

# Function to check for successful landing
def check_landing(lander_x, lander_y, lander_vel_x, lander_vel_y, angle, pad_x, pad_y, pad_width):
    # Define landing conditions
    velocity_threshold = 2.0  # Maximum velocity for a safe landing
    angle_threshold = 15.0  # Maximum angle for a safe landing (in degrees)

    # Check if lander is within the landing pad's horizontal bounds
    if (lander_x + lander_width >= pad_x and lander_x <= pad_x + pad_width):
        # Check if lander is close to the landing pad vertically
        if lander_y + lander_height >= pad_y:
            # Check if velocity is within the threshold
            if abs(lander_vel_x) < velocity_threshold and abs(lander_vel_y) < velocity_threshold:
                # Check if lander is roughly upright
                if abs(angle) < angle_threshold:
                    return "success"
                else:
                    return "crash (angle too steep)"
            else:
                return "crash (too fast)"
    return None

# Game loop
clock = pygame.time.Clock()
running = True

while running:
    screen.fill(BLACK)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Key handling
    keys = pygame.key.get_pressed()
    thrust_active = False
    if keys[pygame.K_LEFT]:
        angle += rotation_speed  # Rotate counterclockwise
    if keys[pygame.K_RIGHT]:
        angle -= rotation_speed  # Rotate clockwise
    if keys[pygame.K_UP]:
        # Apply thrust in the direction the lander is facing
        radians = math.radians(angle)
        lander_vel_x += thrust * math.sin(radians)
        lander_vel_y -= thrust * math.cos(radians)
        thrust_active = True

    # Apply gravity
    lander_vel_y += gravity

    # Apply damping to velocity
    lander_vel_x *= damping
    lander_vel_y *= damping

    # Update lander position
    lander_x += lander_vel_x
    lander_y += lander_vel_y

    # Draw lander
    draw_lander(screen, lander_x, lander_y, angle)

    # Draw landing pad
    pygame.draw.rect(screen, GREEN, (pad_x, pad_y, pad_width, pad_height))

    # Draw debug information
    draw_debug_info(screen, lander_x, lander_y, lander_vel_x, lander_vel_y, angle, gravity, thrust_active)

    # Check for landing
    landing_result = check_landing(lander_x, lander_y, lander_vel_x, lander_vel_y, angle, pad_x, pad_y, pad_width)
    if landing_result:
        if landing_result == "success":
            print("Successful landing!")
        else:
            print(landing_result)
        running = False

    # Check for out of bounds
    if lander_y > HEIGHT or lander_x < 0 or lander_x + lander_width > WIDTH:
        print("Out of bounds!")
        running = False

    # Update the display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()