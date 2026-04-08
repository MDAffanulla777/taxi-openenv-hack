import gradio as gr
import numpy as np
from taxi_env import TaxiEnv
import json

def run_episode(grid_size, max_steps, seed):
    env = TaxiEnv(size=int(grid_size))
    obs, _ = env.reset(seed=int(seed))

    frames = []
    total_reward = 0
    final_state = None

    for step in range(int(max_steps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        total_reward += reward
        final_state = info
        if terminated or truncated:
            break

    env.close()
    return frames[-1], total_reward, json.dumps(final_state, indent=2)


# -------------------------------------------------------
#  CUSTOM PROFESSIONAL CSS (CITY BACKGROUND + TAXI THEME)
# -------------------------------------------------------
custom_css = """
/* Background Image */
body {
    background: url('https://images.unsplash.com/photo-1465447142348-e9952c393450?q=80&w=2070&auto=format&fit=crop')
        no-repeat center center fixed !important;
    background-size: cover !important;
}

/* Semi-transparent working area */
.gr-block.gr-box {
    border-radius: 15px !important;
    border: 1px solid rgba(255, 204, 0, 0.25) !important;
    background: rgba(0, 0, 0, 0.55) !important;
    backdrop-filter: blur(4px);
}

/* Title */
#title {
    font-size: 48px !important;
    font-weight: 900 !important;
    color: #FFD500 !important; /* Taxi Yellow */
    text-align: center !important;
    margin-bottom: 10px !important;
    text-shadow: 0px 0px 8px rgba(0,0,0,0.65);
}

/* Headings */
h1, h2, h3, label, p {
    color: white !important;
    font-weight: 600;
}

/* Card styling */
.card {
    background: rgba(0, 0, 0, 0.65) !important;
    padding: 20px !important;
    border-radius: 15px !important;
    border: 1px solid rgba(255, 204, 0, 0.4) !important;
    box-shadow: 0px 0px 10px rgba(255, 221, 0, 0.3);
}

/* Buttons */
button {
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 18px !important;
    background: #FFD500 !important; /* Taxi yellow */
    color: black !important;
    font-weight: 800 !important;
}

/* Image Border */
#final-frame img {
    border-radius: 12px !important;
    border: 3px solid #FFD500 !important;
}
"""


# -------------------------------------------------------
#  UI LAYOUT
# -------------------------------------------------------
with gr.Blocks(title="Taxi OpenEnv - Scaler Hackathon", css=custom_css) as demo:

    # CLEAN TITLE ONLY — NO SUBTITLE
    gr.HTML("""
    <h1 id='title'>🚖 Taxi Pickup Environment</h1>
    """)

    with gr.Row():
        with gr.Column(elem_classes=["card"]):
            gr.Markdown("### ⚙️ Environment Settings")

            grid_size = gr.Slider(5, minimum=5, maximum=10, step=1, value=5, label="Grid Size")
            max_steps = gr.Slider(50, minimum=50, maximum=200, step=10, value=100, label="Max Steps")
            seed = gr.Slider(0, minimum=0, maximum=100, step=1, value=42, label="Random Seed")

            run_btn = gr.Button("Run Random Agent")

        with gr.Column(elem_classes=["card"]):
            gr.Markdown("### 🖼️ Final State Output")
            final_frame = gr.Image(
                label="Final Frame (Green=Taxi, Red=Passenger, Blue=Destination)",
                elem_id="final-frame",
                height=400
            )
            score_out = gr.Number(label="Total Reward")

    with gr.Column(elem_classes=["card"]):
        gr.Markdown("### 🧠 ReAct JSON State")
        api_state = gr.Textbox(label="State (JSON)", lines=10)

    run_btn.click(
        run_episode,
        inputs=[grid_size, max_steps, seed],
        outputs=[final_frame, score_out, api_state]
    )

    gr.Markdown("<h2 style='text-align:center; color:#FFD500; margin-top:30px;'>Ready for Deployment</h2>")


if __name__ == "__main__":
    demo.launch()