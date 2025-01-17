import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
DINO_WIDTH, DINO_HEIGHT = 40, 40
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 20, 40
GROUND_HEIGHT = 300
FONT_SIZE = 24
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
FPS = 60

class DinoGame(gym.Env):
    def __init__(self):
        super(DinoGame, self).__init__()

        # Define action and observation space
        # Action space: 0 -> Do nothing, 3 -> Jump
        self.action_space = spaces.Discrete(2)

        # Observation space: [dino_y, obstacle_x, obstacle_speed, score]
        # The values range from 0 to SCREEN_WIDTH for x coordinates and 0 to SCREEN_HEIGHT for y coordinates
        self.observation_space = spaces.Box(low=0, high=np.array([SCREEN_HEIGHT, SCREEN_WIDTH, 10, np.inf]), dtype=np.float32)

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Google Dino Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.dino_y = GROUND_HEIGHT - DINO_HEIGHT  # Dino starts at ground level
        self.dino_velocity = 0
        self.is_jumping = False
        self.obstacle_x = SCREEN_WIDTH
        self.obstacle_speed = 5
        self.reward = 0

        # Return the initial state
        self.state = np.array([self.dino_y, self.obstacle_x, self.obstacle_speed, self.reward], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        """ Takes in an action (int) and returns the next state, reward, and if the game is done """

        # Handle jumping action: only jump if on the ground
        if action == 1 and not self.is_jumping:
            self.is_jumping = True
            self.dino_velocity = -15     # Initial jump velocity

        # Update dino's position if jumping
        if self.is_jumping:
            self.dino_y += self.dino_velocity
            self.dino_velocity += 1  # Gravity effect
            if self.dino_y >= GROUND_HEIGHT - DINO_HEIGHT:
                self.dino_y = GROUND_HEIGHT - DINO_HEIGHT
                self.is_jumping = False

        # Update obstacle position
        self.obstacle_x -= self.obstacle_speed
        if self.obstacle_x < 0:
            self.obstacle_x = SCREEN_WIDTH  # Reset obstacle to the right
            self.reward += 1  # Increment score

        # Collision detection
        done = False
        if self.dino_y + DINO_HEIGHT > GROUND_HEIGHT - OBSTACLE_HEIGHT and self.obstacle_x < 50 + DINO_WIDTH:
            # Check if the dino collides with the obstacle
            done = True
            self.reward -= 5  # Penalty for collision

        # Update state
        self.state = np.array([self.dino_y, self.obstacle_x, self.obstacle_speed, self.reward], dtype=np.float32)

        # Return state, reward, done flag, and additional info
        return self.state, self.reward, done, False, {}

    def render(self, mode="human"):
        """ Renders the game on the screen """
        self.screen.fill(WHITE)

        # Draw the ground line
        pygame.draw.line(self.screen, BLACK, (0, GROUND_HEIGHT), (SCREEN_WIDTH, GROUND_HEIGHT), 2)

        # Draw the dino
        pygame.draw.rect(self.screen, BLACK, (50, self.dino_y, DINO_WIDTH, DINO_HEIGHT))

        # Draw the obstacle
        pygame.draw.rect(self.screen, RED, (self.obstacle_x, GROUND_HEIGHT - OBSTACLE_HEIGHT, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

        # Render the score
        score_text = self.font.render(f"Score: {self.reward}", True, BLACK)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

        return self.screen

    def close(self):
        """ Close the pygame window """
        pygame.quit()


