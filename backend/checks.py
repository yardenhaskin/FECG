import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str, allow_error_codes=(0, 1)):
    print(f"\n🔧 Running {description}...")
    result = subprocess.run(command, cwd=Path(__file__).parent)

    if result.returncode not in allow_error_codes:
        print(f"❌ {description} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    elif result.returncode == 1:
        print(f"⚠️  {description} found issues (exit code 1).")
    else:
        print(f"✅ {description} completed successfully.")


def main():
    # Optional: install tools if not already installed
    # run_command(["uv", "pip", "install", "mypy", "ruff", "black"], "Installing tools")

    # Commands to run (note: no trailing space in args like `" ."` — split them!)
    commands = [
        (["uvx", "ruff", "format", "."], "Ruff formatting check"),
        (["uvx", "black", "."], "Black formatting"),
        (["uvx", "mypy", "."], "Mypy type checking"),
    ]

    for cmd, desc in commands:
        run_command(cmd, desc)


if __name__ == "__main__":
    main()
