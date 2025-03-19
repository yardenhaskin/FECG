import subprocess


def build_exe():
    # Set the command to run PyInstaller
    command = [
        "PyInstaller", "--onefile", "--noconsole", "--icon=server_115155.ico",
        "--add-data", "requirements-true.txt;.",
        "--add-data", "requirements-false.txt;.",
        "--add-data", "docker-compose.cpu.yml;.",
        "--add-data", "docker-compose.gpu.yml;.",
        "--add-data", "requirements.txt;.",
        "--add-data", "Dockerfile;.",
        "--add-data", ".dockerignore;.",
        "-n", "run_FECG_server", ".\\run_docker.py"
    ]

    try:
        # Run the PyInstaller command
        subprocess.check_call(command)
        print("Build completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during build: {e}")


if __name__ == "__main__":
    build_exe()
