import os
import sys
import platform
import logging
from pathlib import Path
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlatformLauncher:
    """Platform-specific launcher for AI Scientist."""

    def __init__(self):
        self.platform = platform.system()
        self.python_path = sys.executable
        self.base_dir = Path(__file__).parent.parent.absolute()

    def setup_environment(self) -> Dict[str, str]:
        """Set up environment variables based on platform."""
        env = os.environ.copy()

        if self.platform == "Windows":
            # Convert paths to Windows format
            env["PYTHONPATH"] = str(self.base_dir).replace("/", "\\")
            # Add Windows-specific environment variables
            env["APPDATA"] = os.path.expandvars("%APPDATA%")
            env["LOCALAPPDATA"] = os.path.expandvars("%LOCALAPPDATA%")
        else:
            env["PYTHONPATH"] = str(self.base_dir)

        return env

    def get_python_command(self) -> str:
        """Get the appropriate Python command for the platform."""
        if self.platform == "Windows":
            return str(Path(self.python_path).resolve())
        return self.python_path

    def launch(self, script_path: str, args: Optional[list] = None) -> int:
        """Launch a Python script with platform-specific configuration."""
        try:
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Script not found: {script_path}")

            env = self.setup_environment()
            python_cmd = self.get_python_command()

            # Build command list
            cmd = [python_cmd, script_path]
            if args:
                cmd.extend(args)

            logger.info(f"Launching on {self.platform} platform")
            logger.info(f"Command: {' '.join(cmd)}")

            # Use the appropriate module based on platform
            if self.platform == "Windows":
                from subprocess import CREATE_NO_WINDOW
                import subprocess
                return subprocess.run(
                    cmd,
                    env=env,
                    creationflags=CREATE_NO_WINDOW,
                    check=True
                ).returncode
            else:
                import subprocess
                return subprocess.run(
                    cmd,
                    env=env,
                    check=True
                ).returncode

        except FileNotFoundError as e:
            logger.error(f"Launch failed: {e}")
            return 1
        except subprocess.CalledProcessError as e:
            logger.error(f"Process failed with return code {e.returncode}")
            return e.returncode
        except Exception as e:
            logger.error(f"Unexpected error during launch: {e}")
            return 1

def main():
    """Main entry point for the launcher."""
    launcher = PlatformLauncher()

    # Example usage
    if len(sys.argv) < 2:
        logger.error("Usage: python launcher.py <script_path> [args...]")
        return 1

    script_path = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else None

    return launcher.launch(script_path, args)

if __name__ == "__main__":
    sys.exit(main())
