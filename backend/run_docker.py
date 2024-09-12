#!/usr/bin/env python

import subprocess
import ctypes


def check_docker_running():
    try:
        subprocess.check_call(['docker', 'info'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def show_error_message(message):
    ctypes.windll.user32.MessageBoxW(0, message, "Error", 0x10)


def run_docker():
    if not check_docker_running():
        show_error_message("Docker is not running. Please start Docker and try again.")
        return

    try:
        # Build the Docker image
        subprocess.check_call(['docker-compose', 'build'])
        # Run the Docker container with --remove-orphans flag
        subprocess.check_call(['docker-compose', 'up', '--remove-orphans'])
    except subprocess.CalledProcessError as e:
        show_error_message(f"An error occurred: {e}")


if __name__ == "__main__":
    run_docker()
