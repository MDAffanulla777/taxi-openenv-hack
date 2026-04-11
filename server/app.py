from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from taxi_env import TaxiEnv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = TaxiEnv(size=5)

class StepInput(BaseModel):
    action: int


@app.get("/")
def home():
    return {"status": "OK"}


@app.post("/reset")
def reset():
    obs, info = env.reset()
    return {"obs": obs.tolist(), "info": info}


@app.post("/step")
def step(data: StepInput):
    obs, reward, terminated, truncated, info = env.step(data.action)
    return {
        "obs": obs.tolist(),
        "reward": float(reward),
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }


# 🔥 REQUIRED BY OPENENV
def main():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
