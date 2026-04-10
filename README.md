---
title: TaxiEnv Demo
emoji: 🚖
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_file: app.py
---
🚖 TaxiEnv – Custom Open Environment (Scaler Hackathon)
This project implements a custom Taxi environment compatible with OpenAI Gym–style APIs and exposed using FastAPI.
It is deployed on HuggingFace Spaces for evaluation.

🚀 Live API Demo
🔗 HuggingFace Space:
https://mohammedaffanulla-affanulla-taxi-env-demo.hf.space

🎥 Demo Video
🔗 Demo Video Link (Google Drive):
https://drive.google.com/file/d/1w2KK9DRihhMmKFdDeKBE0E1YTJpzZ3qe/view?usp=drive_link

🧪 API Endpoints (Required for Hackathon)
✅ POST /reset
Resets the environment and returns the initial observation.

Example Response:

{
  "observation": [...],
  "info": {}
}
✅ POST /step

Executes one step in the environment.

Example Request:

{
  "action": 0
}

Example Response:

{
  "observation": [...],
  "reward": -1,
  "terminated": false,
  "truncated": false,
  "info": {
    "taxi": [2,2],
    "passenger": [4,0],
    "destination": [0,4],
    "steps": 1
  }
}
🧠 Environment Features
Grid size: 5 × 5
Observation: RGB matrix (5×5×3)
Objects:
🟩 Taxi
🟥 Passenger
🟦 Destination
Rewards:
+10 pickup
+20 drop-off
−5 invalid pickup/drop
−1 per step
Episode length: 200 steps
📁 Project Structure
├── app.py
├── taxi_env.py
├── requirements.txt
├── Dockerfile
└── README.md
⚙️ Running Locally
Install dependencies:
pip install -r requirements.txt
Run API:
python app.py

Runs on → http://localhost:7860

🐳 Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]
✔️ Hackathon Checklist
Requirement	Status
Custom environment implemented	✅
/reset endpoint works	✅
/step endpoint works	✅
OpenEnv compatible	✅
HuggingFace Space running	✅
Demo video added	✅
README included	✅
👥 Team Name

TRIDENT

👥 Team Members
Mohammed Affanulla
Arbain Hussain
Prajwal K
