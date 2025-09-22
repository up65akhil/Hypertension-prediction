# Use a slim, official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Define the command to run your application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
