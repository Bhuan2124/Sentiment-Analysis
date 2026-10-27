"""
Create the project-local virtual environment (.venv) and install everything the dashboard, the tests and
the evaluation script need. Safe to run again: it only adds what is missing.

    python scripts/setup_env.py

Everything is written to the terminal and to logs/setup_env.log. The environment is isolated, so the
packages already installed for your notebooks are not touched. Versions of numpy / pandas / scikit-learn
are pinned to the ones used for training, so the saved models load without warnings.
"""
import os
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "setup_env.log"
VENV = ROOT / ".venv"


def log(message: str = "") -> None:
    print(message, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in proc.stdout:
        log(line.rstrip())
    return proc.wait()


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Setup started {time.strftime('%Y-%m-%d %H:%M:%S')} with Python {sys.version.split()[0]} ({sys.executable})")

    if not venv_python().is_file():
        log("Creating .venv ...")
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    else:
        log(".venv already exists.")
    py = str(venv_python())

    run([py, "-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check", "--no-input"])

    code = run([py, "-m", "pip", "install", "-r", "requirements-app.txt", "--disable-pip-version-check", "--no-input"])
    if code != 0:
        log("\nThe pinned versions could not be installed together. Trying the relaxed list (see the messages above).")
        code = run([py, "-m", "pip", "install", "-r", "requirements-app-relaxed.txt", "--disable-pip-version-check", "--no-input"])
    if code != 0:
        log("\nSETUP FAILED while installing requirements. Read the pip messages above.")
        return code

    # PyTorch: the CPU build is enough for the dashboard (DistilBERT answers one review at a time)
    code = run([py, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
                "--disable-pip-version-check", "--no-input"])
    if code != 0:
        log("\nThe CPU build of torch was not found. Trying the default build from PyPI.")
        code = run([py, "-m", "pip", "install", "torch", "--disable-pip-version-check", "--no-input"])
    if code != 0:
        log("\nSETUP FAILED while installing torch.")
        return code

    check = ("import importlib.metadata as m, sys\n"
             "names=['streamlit','plotly','pandas','numpy','scikit-learn','joblib','matplotlib','pytest','torch','transformers']\n"
             "print('python', sys.version.split()[0])\n"
             "print(*[f'{n} {m.version(n)}' for n in names], sep='\\n')\n"
             "import streamlit, plotly, sklearn, torch, transformers\n"
             "print('IMPORT CHECK OK')\n")
    code = run([py, "-c", check])
    log("\nSETUP FINISHED" if code == 0 else "\nSETUP FAILED the import check. Read the messages above.")
    return code


if __name__ == "__main__":
    sys.exit(main())
