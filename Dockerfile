# Step 1: Use a slim, official Python base image
# This provides a lightweight environment with Python pre-installed.
FROM python:3.11-slim

# Step 2: Set the working directory inside the container to /app
# All subsequent commands will run from this directory.
WORKDIR /app

# Step 3: Copy only the requirements file first
# This is a Docker caching optimization. If this file doesn't change,
# the next step can be skipped in future builds, making them faster.
COPY requirements.txt .

# Step 4: Install all Python packages from the requirements file
# --no-cache-dir keeps the final image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all of your project files into the container
# This includes app.py, the .pkl model, and the .csv dataset.
COPY . .

# Step 6: Expose the port that Gunicorn will run on
EXPOSE 8000

# Step 7: Define the command to start the application server
# This runs your Flask app using Gunicorn, a production-ready server.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
