"""Drive the car in 3D: starts the Python physics server and the Godot front end together.

    python drive3d.py                                   # test circuit, dry, you drive
    python drive3d.py --track hill --weather shower --compound intermediate
    python drive3d.py --ai                              # watch the AI drive the racing line

Needs Godot 4.3 or newer (https://godotengine.org/download). Put it on your PATH as `godot`,
or point the GODOT environment variable at the executable. Without Godot, the server still
starts: open godot/project.godot in the Godot editor and press F5.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GODOT_DIR = ROOT / "godot"


def find_godot():
    if os.environ.get("GODOT"):
        return os.environ["GODOT"]
    for name in ("godot", "godot4", "Godot", "godot-4", "Godot_v4.3-stable_linux.x86_64"):
        path = shutil.which(name)
        if path:
            return path
    mac = "/Applications/Godot.app/Contents/MacOS/Godot"
    return mac if os.path.exists(mac) else None


def main():
    server = subprocess.Popen([sys.executable, "-m", "carsim.server", *sys.argv[1:]], cwd=ROOT)
    godot = find_godot()
    try:
        if godot is None:
            print("\nGodot not found. The physics server is running: open godot/project.godot in the Godot\n"
                  "editor (4.3 or newer) and press F5. Or set GODOT=/path/to/godot and run this again.\n"
                  "Press Ctrl+C here to stop the server.")
            server.wait()
            return
        if not (GODOT_DIR / ".godot").exists():
            print("First run: Godot is importing the project ...")
            subprocess.run([godot, "--headless", "--path", str(GODOT_DIR), "--import"], check=False)
        subprocess.run([godot, "--path", str(GODOT_DIR)], check=False)
    except KeyboardInterrupt:
        pass
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
