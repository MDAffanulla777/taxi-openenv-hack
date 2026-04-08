# Use official lightweight Python image
FROM python:3.10-slim

# Avoid Python writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install OS dependencies (if needed later)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
# If you have a requirements.txt, uncomment the next line:
RUN pip install -r requirements.txt

# Expose port for Gradio
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
