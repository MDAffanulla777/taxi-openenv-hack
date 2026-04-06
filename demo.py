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
    return frames[-1], total_reward, json.dumps(final_state, indent=2)  # Last frame + score + JSON string

with gr.Blocks(title="🚖 Taxi OpenEnv - Scaler Hackathon") as demo:
    gr.Markdown("# 🚖 Taxi Pickup Environment\n**Custom Gymnasium Open World Env for ReAct Agents**")
    gr.Markdown("- Partial fog observation\n- 6 discrete actions (NSEW+Pickup+Dropoff)\n- JSON state API ready\n- Baseline random agent")
    
    with gr.Row():
        with gr.Column():
            grid_size = gr.Slider(5, minimum=5, maximum=10, step=1, value=5, label="Grid Size")
            max_steps = gr.Slider(50, minimum=50, maximum=200, step=10, value=100, label="Max Steps")
            seed = gr.Slider(0, minimum=0, maximum=100, step=1, value=42, label="Seed")
            run_btn = gr.Button("🚀 Run Random Agent", variant="primary")
    
    with gr.Row():
        final_frame = gr.Image(label="Final Frame")
        score_out = gr.Number(label="Total Reward")
    
    api_state = gr.Textbox(label="ReAct API State (JSON)", lines=8)
    
    run_btn.click(
        run_episode, 
        inputs=[grid_size, max_steps, seed], 
        outputs=[final_frame, score_out, api_state]
    )
    
    gr.Markdown("## Ready for HF Spaces & Submission!")

if __name__ == "__main__":
    demo.launch()
