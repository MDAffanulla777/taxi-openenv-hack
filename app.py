from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from taxi_env import TaxiEnv

app = FastAPI(
    title="TaxiEnv API",
    description="Scaler Hackathon OpenEnv Compatible API",
    version="1.0.0"
)

# Allow CORS
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
    return {"status": "OK", "message": "TaxiEnv API Running"}


@app.post("/reset")
def reset_env():
    obs, info = env.reset()
    return {"obs": obs.tolist(), "info": info}


@app.post("/step")
def step_env(data: StepInput):
    obs, reward, terminated, truncated, info = env.step(data.action)
    return {
        "obs": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }


# ✅ IMPORTANT: Hackathon entry fix
def main():
    return app
