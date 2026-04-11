from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from taxi_env import TaxiEnv
import uvicorn

app = FastAPI(
    title="TaxiEnv API",
    description="Scaler Hackathon OpenEnv Compatible API",
    version="1.0.0"
)

# CORS (required for HF + web UI calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize environment
env = TaxiEnv(size=5)


# Input schema
class StepInput(BaseModel):
    action: int


# Health check route
@app.get("/")
def home():
    return {"status": "OK", "message": "TaxiEnv API Running"}


# Reset environment
@app.post("/reset")
def reset_env():
    obs, info = env.reset()
    return {
        "obs": obs.tolist(),
        "info": info
    }


# Step function
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


# HF + local run support
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
