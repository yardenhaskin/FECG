# Use the official Python 3.10 image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install the dependencies
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables
ENV FLASK_APP="server:create_app"
ENV FLASK_ENV=development

# Define the command to run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
# Testing threads on model
#CMD ["python", "/app/server/routes/__init__.py"]
