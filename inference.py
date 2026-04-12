from taxi_env import TaxiEnv

def run():
    env = TaxiEnv(size=5)

    # START block
    print("[START] task=taxi", flush=True)

    obs, info = env.reset()

    total_reward = 0
    steps = 0

    for i in range(5):  # simple demo loop
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        print(f"[STEP] step={steps} reward={reward}", flush=True)

        if terminated or truncated:
            break

    score = total_reward

    # END block
    print(f"[END] task=taxi score={score} steps={steps}", flush=True)


if __name__ == "__main__":
    run()
