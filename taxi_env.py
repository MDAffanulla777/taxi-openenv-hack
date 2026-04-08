import numpy as np
import random

class TaxiEnv:
    def __init__(self, size=5, max_steps=200):
        self.size = size
        self.max_steps = max_steps
        self.action_space = type("space", (), {"sample": lambda self: random.randint(0, 5)})()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0

        # random taxi, passenger, destination
        self.taxi_pos = (
            np.random.randint(0, self.size),
            np.random.randint(0, self.size),
        )
        self.passenger_pos = (
            np.random.randint(0, self.size),
            np.random.randint(0, self.size),
        )
        self.destination = (
            np.random.randint(0, self.size),
            np.random.randint(0, self.size),
        )

        self.passenger_in_taxi = False
        return self._get_obs(), {}

    def _get_obs(self):
        # Observation is the grid encoded (0 empty, 1 taxi, 2 passenger, 3 destination)
        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        tx, ty = self.taxi_pos
        grid[tx, ty] = [0, 255, 0]  # Green taxi

        if not self.passenger_in_taxi:
            px, py = self.passenger_pos
            grid[px, py] = [255, 0, 0]  # Red passenger

        dx, dy = self.destination
        grid[dx, dy] = [0, 0, 255]  # Blue destination

        return grid

    def step(self, action):
        x, y = self.taxi_pos

        # movement
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.size - 1:
            y += 1

        self.taxi_pos = (x, y)
        reward = -1
        terminated = False

        # pickup
        if action == 4:
            if self.taxi_pos == self.passenger_pos and not self.passenger_in_taxi:
                self.passenger_in_taxi = True
                reward = +10
            else:
                reward = -5

        # drop
        if action == 5:
            if self.passenger_in_taxi and self.taxi_pos == self.destination:
                reward = +20
                terminated = True
            else:
                reward = -5

        self.steps += 1
        truncated = self.steps >= self.max_steps

        info = {
            "taxi": self.taxi_pos,
            "passenger": self.passenger_pos,
            "destination": self.destination,
            "in_taxi": self.passenger_in_taxi,
            "steps": self.steps,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # FULLY FIXED — ALWAYS RETURNS STRING (Never None)
    def render(self):
        grid = [[" . " for _ in range(self.size)] for _ in range(self.size)]

        tx, ty = self.taxi_pos
        grid[tx][ty] = " T "

        if not self.passenger_in_taxi:
            px, py = self.passenger_pos
            grid[px][py] = " P "

        dx, dy = self.destination
        grid[dx][dy] = " D "

        grid_str = "\n".join(["".join(row) for row in grid])

        return (
            "TaxiEnv Render\n"
            f"Taxi: {self.taxi_pos}\n"
            f"Passenger: {self.passenger_pos} (in taxi: {self.passenger_in_taxi})\n"
            f"Destination: {self.destination}\n"
            f"Steps: {self.steps}/{self.max_steps}\n\n"
            f"{grid_str}\n"
        )
