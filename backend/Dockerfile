## Use the official Python 3.10 image from the Docker Hub
#FROM python:3.10-slim
#
## Set the working directory in the container
#WORKDIR /app
#
## Copy the project files into the container
#COPY . .
#
## Install the dependencies
#RUN pip install -r requirements-true.txt
#
## Expose the port the app runs on
#EXPOSE 5000
#
## Set environment variables
#ENV FLASK_APP="server:create_app"
#ENV FLASK_ENV=development
#
## Define the command to run the application
#CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
# Testing threads on model
#CMD ["python", "/app/server/routes/__init__.py"]

# Stage 1: Base image
FROM python:3.10-alpine3.21 AS base

# Set the working directory
WORKDIR /app

# Set build argument to specify whether to use GPU or CPU
ARG USE_GPU=false

# Upgrade pip first to ensure compatibility with package installations
RUN pip install --upgrade pip

# Conditionally copy the appropriate requirements file
COPY requirements-${USE_GPU}.txt /app/requirements.txt

# Install the dependencies based on the selected requirements file
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose port for the application
EXPOSE 5000

# Set environment variables
ENV FLASK_APP="server:create_app"
ENV FLASK_ENV=development

# Define the command to run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]

# Stage 2: Use a clean, lightweight image for runtime (if you need to use a virtual environment)
# Use the same image as the base stage for simplicity in this case
FROM python:3.10-alpine3.21 AS runtime

# Set the working directory
WORKDIR /app

# Copy necessary files from the base stage
COPY --from=base /app /app

# Expose the port again in case it is overridden in the previous stage
EXPOSE 5000

# Define the command to run the application again
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]


