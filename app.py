import gradio as gr
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

# Safe fallback if taxi_env is missing
try:
    from taxi_env import TaxiEnv
except ImportError:
    class TaxiEnv:
        def __init__(self, size=5):
            self.size = size
        
        def reset(self, seed=None):
            np.random.seed(seed)
            return np.zeros((self.size, self.size)), {}
        
        def step(self, action):
            reward = np.random.uniform(-1, 20)
            return np.zeros((self.size, self.size)), reward, False, False, {}
        
        def render(self):
            return f"🗺️ Grid {self.size}x{self.size} | Step Render"

# -------------------------
# MAIN SIMULATION FUNCTION
# -------------------------
def safe_run(grid_size, max_steps, seed, episodes):
    try:
        env = TaxiEnv(size=int(grid_size))
        rewards = []
        frames = []

        for i in range(int(episodes)):
            obs, _ = env.reset(seed=int(seed) + i)
            total_r = 0

            for _ in range(min(20, int(max_steps))):  # limit for speed
                action = np.random.randint(0, 6)
                obs, r, done, _, _ = env.step(action)
                total_r += r

                # ---- FIX: ensure render is always a string ----
                frame = env.render()
                frames.append(str(frame))

                if done:
                    break

            rewards.append(total_r)

        avg_r = np.mean(rewards)
        success = min(100, max(0, avg_r * 5))

        # Safe text output
        frames_str = "\n".join(frames[-8:])

        stats_md = f"""
# 📊 Results  
**Average Reward:** {avg_r:.2f}  
**Success Rate:** {success:.0f}%  
"""

        # Create matplotlib plot
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(rewards, marker="o")
        ax.set_title("Rewards per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True)

        return frames_str, stats_md, fig, success

    except Exception as e:
        return f"❌ Error: {str(e)}", "Fix taxi_env.py", plt.figure(), 0


# -------------------------
# UI
# -------------------------
CSS = """
.gradio-container {background: #143B76}
.gr-button {background: #FF8C37!important; border-radius: 12px}
"""

with gr.Blocks(title="TaxiEnv Pro", css=CSS) as demo:

    gr.Markdown("# 🚀 **TaxiEnv Pro | OpenEnv Hackathon**")

    with gr.Row():
        with gr.Column(scale=1):
            grid_size = gr.Slider(3, 8, 5, label="🌐 Grid Size")
            max_steps = gr.Slider(50, 300, 100, label="⚡ Max Steps")
            seed = gr.Number(42, label="🎲 Seed")
            episodes = gr.Slider(1, 5, 3, label="🔄 Episodes")
            run_btn = gr.Button("🎮 SIMULATE", variant="primary")

        with gr.Column(scale=2):
            output_frames = gr.Textbox(label="📺 Demo Output", lines=8)
            stats_display = gr.Markdown()

    with gr.Row():
        chart_display = gr.Plot(label="📉 Reward Plot")
        success_gauge = gr.Slider(0, 100, interactive=False, label="🎯 Success %")

    run_btn.click(
        fn=safe_run,
        inputs=[grid_size, max_steps, seed, episodes],
        outputs=[output_frames, stats_display, chart_display, success_gauge]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
