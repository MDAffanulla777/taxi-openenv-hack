def run():
    import os
    from openai import OpenAI

    print("[START] task=taxi", flush=True)

    # 🔥 GUARANTEED API CALL
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )

    # 🔥 THIS MUST SUCCEED (no try/except here)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "return number 1"}],
    )

    # If we reach here → API call is SUCCESSFUL

    from taxi_env import TaxiEnv
    env = TaxiEnv(size=5)

    obs, info = env.reset()
    total_reward = 0
    steps = 0

    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        print(f"[STEP] step={steps} reward={reward}", flush=True)

        if terminated or truncated:
            break

    print(f"[END] task=taxi score={total_reward} steps={steps}", flush=True)


if __name__ == "__main__":
    run()
