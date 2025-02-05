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
#RUN pip install -r requirements.txt
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
FROM python:3.10-slim AS base

# Set the working directory
WORKDIR /app

# Set build argument to specify whether to use GPU or CPU
ARG USE_GPU=false

# Upgrade pip first to ensure compatibility with package installations
RUN pip install --upgrade pip

# Conditionally copy the appropriate requirements file
COPY requirements-true.txt /app/requirements-true.txt
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements-true.txt

# Expose port for the application (optional in base stage)
EXPOSE 5000

# Stage 2: Runtime image
FROM python:3.10-alpine3.21 AS runtime

# Set the working directory
WORKDIR /app

# Copy the necessary files, including the installed site packages from the base image
COPY --from=base /app /app
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Clean up unnecessary build dependencies (adjust according to what you installed in the base image)
RUN rm -rf /root/.cache \
    && apk del --no-cache gcc musl-dev libffi-dev

# Expose the port for the application
EXPOSE 5000

# Define the command to run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]


