"""
Start the Streamlit dashboard and open it in your browser.

    python scripts/run_app.py             # picks port 8501 (or the next free one) and opens the browser
    python scripts/run_app.py --no-browser

This is the same as   python -m streamlit run app.py   plus a free-port check and a log of the address
(logs/app_url.txt). Press Ctrl+C in the terminal to stop the server.
"""
import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def venv_python() -> str:
    candidate = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(candidate) if candidate.is_file() else sys.executable


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    port = args.port
    while port_is_open(port) and port < args.port + 20:
        port += 1
    url = f"http://localhost:{port}"
    cmd = [venv_python(), "-m", "streamlit", "run", "app.py", "--server.port", str(port),
           "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT)

    for _ in range(120):
        if proc.poll() is not None:
            print("The server stopped before it was ready. Read the messages above.", flush=True)
            return proc.returncode or 1
        if port_is_open(port):
            break
        time.sleep(0.5)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "app_url.txt").write_text(url + "\n", encoding="utf-8")
    print(f"\nDASHBOARD RUNNING at {url}   (press Ctrl+C here to stop)\n", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    sys.exit(main())
