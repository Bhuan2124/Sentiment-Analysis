"""
Run all automated tests and save the report to results/test_report.txt.

    python scripts/run_tests.py

Uses the project's .venv if it exists. Falls back to unittest when pytest is not installed.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "test_report.txt"


def venv_python() -> str:
    candidate = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(candidate) if candidate.is_file() else sys.executable


def main() -> int:
    py = venv_python()
    has_pytest = subprocess.run([py, "-c", "import pytest"], capture_output=True).returncode == 0
    if has_pytest:
        cmd = [py, "-m", "pytest", "tests", "-v", "--tb=short", "-p", "no:cacheprovider"]
    else:
        cmd = [py, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]
    header = [f"Test run {time.strftime('%Y-%m-%d %H:%M:%S')}", f"Python: {py}", "$ " + " ".join(cmd), ""]
    print("\n".join(header), flush=True)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = list(header)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))
    code = proc.wait()
    lines.append("")
    lines.append("RESULT: ALL TESTS PASSED" if code == 0 else f"RESULT: TESTS FAILED (exit code {code})")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-1], flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
