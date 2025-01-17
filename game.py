from env import DinoGame
import pygame
if __name__ == "__main__":
    env = DinoGame()
    done = False
    obs, _ = env.reset()

    # Instructions
    print("Press SPACE to jump. Close the game window to exit.")

    while not done:
        env.render()  # Render the environment to the screen

        action = 0  # Default action is to do nothing

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True  # Exit if the game window is closed
            elif event.type == pygame.KEYDOWN:
                # When the SPACE key is pressed, set action to 1 (jump)
                if event.key == pygame.K_SPACE:
                    action = 1 # Action 1 corresponds to jumping

        # Take a step in the environment
        obs, reward, done, truncated, info = env.step(action)

    env.close()  

    