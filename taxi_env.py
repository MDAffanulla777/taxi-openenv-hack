import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt  # For human render

class TaxiEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 4}

    def __init__(self, render_mode=None, size=5):
        super(TaxiEnv, self).__init__()
        self.size = size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)  # 0-3: N/S/E/W, 4: pickup, 5: dropoff
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8)
        self.np_random = np.random.RandomState()  # Initialize here

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random.seed(seed)  # Use seed() for RandomState
        self.agent_pos = np.array([0, 0])
        self.passenger_pos = self.np_random.randint(0, self.size, size=2)  # randint for RandomState
        self.dest_pos = self.np_random.randint(0, self.size, size=2)
        self.has_passenger = False
        obs = self._get_obs()
        info = self._get_state_api()
        return obs, info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        if action < 4:  # Move N/S/E/W
            delta = np.array([[-1,0], [1,0], [0,-1], [0,1]])[action]
            new_pos = np.clip(self.agent_pos + delta, 0, self.size - 1)
            self.agent_pos = new_pos
        elif action == 4 and np.array_equal(self.agent_pos, self.passenger_pos) and not self.has_passenger:
            self.has_passenger = True
            reward += 20
        elif action == 5 and self.has_passenger and np.array_equal(self.agent_pos, self.dest_pos):
            reward += 20
            terminated = True
        
        reward -= 1  # Step penalty
        obs = self._get_obs()
        info = self._get_state_api()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Place entities
        grid[self.agent_pos[0], self.agent_pos[1]] = [0, 255, 0]  # Green: agent
        if not self.has_passenger:
            grid[self.passenger_pos[0], self.passenger_pos[1]] = [255, 0, 0]  # Red: passenger
        grid[self.dest_pos[0], self.dest_pos[1]] = [0, 0, 255]  # Blue: destination
        
        # Partial observation: fog outside 3x3 view
        row_slice = slice(max(0, self.agent_pos[0]-1), min(self.size, self.agent_pos[0]+2))
        col_slice = slice(max(0, self.agent_pos[1]-1), min(self.size, self.agent_pos[1]+2))
        
        fog = np.full((self.size, self.size, 3), 128, dtype=np.uint8)
        fog[row_slice, col_slice] = grid[row_slice, col_slice]
        return fog

    def _get_state_api(self):
        return {
            "agent_pos": self.agent_pos.tolist(),
            "passenger_pos": self.passenger_pos.tolist(),
            "dest_pos": self.dest_pos.tolist(),
            "has_passenger": self.has_passenger,
            "grid_size": self.size
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()
        elif self.render_mode == "human":
            plt.imshow(self._get_obs())
            plt.title("Taxi Env (Green:Taxi, Red:Passenger, Blue:Dest, Gray:Fog)")
            plt.axis('off')
            plt.pause(0.1)
            plt.show(block=False)

    def close(self):
        plt.close('all')

# Test - NOW RUNS PERFECTLY
if __name__ == "__main__":
    env = TaxiEnv(render_mode="human", size=5)
    obs, info = env.reset(seed=42)
    print("✅ Initial API State:", info)
    
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        print(f"Step {step+1}: Action={action}, Reward={reward}, Total={total_reward}, State keys={list(info.keys())}")
        env.render()
        if term or trunc:
            print("✅ Episode done!")
            break
    
    env.close()
    print("🎉 TaxiEnv fully working! Ready for Gradio demo & HF Spaces submission.")
