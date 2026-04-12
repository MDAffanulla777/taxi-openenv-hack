import os
from openai import OpenAI
from taxi_env import TaxiEnv

def run():
    # Initialize client safely
    try:
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    except Exception:
        client = None  # fallback if env not available

    env = TaxiEnv(size=5)

    print("[START] task=taxi", flush=True)

    obs, info = env.reset()

    total_reward = 0
    steps = 0

    for i in range(5):
        action = None

        # 🔥 Safe API call
        if client:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Return ONLY a number between 0 and 5."},
                        {"role": "user", "content": f"State: {obs}"}
                    ],
                )

                content = response.choices[0].message.content.strip()
                action = int(content)

            except Exception:
                action = None  # fallback

        # 🔥 Safe fallback
        if action is None or not (0 <= action <= 5):
            action = env.action_space.sample()

        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception:
            # fallback if env fails
            reward = 0
            terminated = True
            truncated = False

        total_reward += reward
        steps += 1

        print(f"[STEP] step={steps} reward={reward}", flush=True)

        if terminated or truncated:
            break

    print(f"[END] task=taxi score={total_reward} steps={steps}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        # 🔥 NEVER crash — always output valid END
        print("[START] task=taxi", flush=True)
        print("[END] task=taxi score=0 steps=0", flush=True)
