# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
# This copies src folder, and potentially other necessary files (e.g., if model is local)
COPY src /app/src
COPY .mlflowignore .
COPY .git /app/.git # Needed if MLflow tries to infer git context
COPY .github /app/.github # Copying workflows for completeness, not strictly needed for runtime

# MLflow related files (if you expect to load from local directory, otherwise not strictly needed for container)
# If your model is in MLflow Registry (as specified in main.py), these aren't critical for runtime.
# If you are serving a local model, you'd COPY the mlflow artifacts folder here.
# COPY mlruns /app/mlruns

# Set environment variables for MLflow if your registry/tracking server is remote
# If you are using a local MLflow server (e.g., in a docker-compose setup), these are crucial.
# Replace with your actual MLflow server URI if different
ENV MLFLOW_TRACKING_URI=http://mlflow_server:5000
ENV MLFLOW_REGISTRY_URI=http://mlflow_server:5000

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn
# The command should point to your main.py within the src/api directory
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]