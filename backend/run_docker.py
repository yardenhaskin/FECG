import subprocess

def run_docker():
    try:
        # Build the Docker image
        subprocess.check_call(['docker-compose', 'build'])
        # Run the Docker container
        subprocess.check_call(['docker-compose', 'up'])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_docker()