from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from taxi_env import TaxiEnv

# 🔥 Rename from app → api (IMPORTANT)
api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = TaxiEnv(size=5)

class StepInput(BaseModel):
    action: int


@api.get("/")
def home():
    return {"status": "OK"}


@api.post("/reset")
def reset():
    obs, info = env.reset()
    return {"obs": obs.tolist(), "info": info}


@api.post("/step")
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
    return api


# 🔥 Required for Docker/local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=7860)
