from fastapi import FastAPI
from pydantic import BaseModel
from taxi_env import TaxiEnv  
import uvicorn

app = FastAPI()

env = TaxiEnv()

class Action(BaseModel):
    action: int

@app.get("/")
def root():
    return {"status": "OK", "message": "TaxiEnv API Running"}

@app.post("/reset")
def reset_env():
    obs = env.reset()
    return {"observation": obs}

@app.post("/step")
def step_env(data: Action):
    obs, reward, done, info = env.step(data.action)

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/render")
def render_env():
    frame = env.render()
    return {"frame": frame}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
