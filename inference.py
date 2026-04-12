import os
from openai import OpenAI
from taxi_env import TaxiEnv

def run():
    # 🔥 Use injected environment variables
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )

    env = TaxiEnv(size=5)

    print("[START] task=taxi", flush=True)

    obs, info = env.reset()

    total_reward = 0
    steps = 0

    for i in range(5):
        # 🔥 Make REQUIRED API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a taxi agent."},
                {"role": "user", "content": f"Current state: {obs}. Give action (0-5)."}
            ],
        )

        # Extract action safely
        try:
            action = int(response.choices[0].message.content.strip())
            action = max(0, min(5, action))
        except:
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
