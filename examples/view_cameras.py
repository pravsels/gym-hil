import argparse
import cv2
import gymnasium as gym
import numpy as np
import time

import gym_hil

def main():
    parser = argparse.ArgumentParser(description="Stream camera views from the ARX5 environment.")
    parser.add_argument("--fps", type=int, default=20, help="Target frames per second for display.")
    args = parser.parse_args()

    # Create the environment with image observations enabled
    env = gym.make("gym_hil/Arx5BlockTowerBase-v0", image_obs=True)
    
    obs, info = env.reset()
    print("Camera streams started. Press 'q' or 'ESC' in the image window to quit.")
    
    # We'll just send dummy actions (stay still) to keep the simulation ticking
    # The action space is 7D (6 arm joints + 1 gripper)
    # We can just read the current position from the observation to stay still
    
    try:
        while True:
            start_time = time.time()
            
            # The images are stored in obs["pixels"] as HWC uint8 numpy arrays
            front_img = obs["pixels"]["front"]
            wrist_img = obs["pixels"]["wrist"]
            
            # OpenCV expects BGR format instead of RGB, so we convert them
            front_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
            wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
            
            # Resize wrist cam to match height of front cam if we want to concatenate them
            # Both are currently 480x640, so we can just put them side by side
            combined_img = np.hstack((front_bgr, wrist_bgr))
            
            # Add some text labels
            cv2.putText(combined_img, "Front Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_img, "Wrist Camera", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("ARX5 Camera Streams", combined_img)
            
            # Wait for key press (1ms) and check if user wants to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC
                break
                
            # Step the environment (dummy action to stay at current state)
            # Use the current state as the target position
            dummy_action = obs["observation.state"] 
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            
            if terminated or truncated:
                obs, info = env.reset()
                
            # Sleep to match target FPS roughly
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / args.fps) - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()
        cv2.destroyAllWindows()
        print("Closed streams.")

if __name__ == "__main__":
    main()
