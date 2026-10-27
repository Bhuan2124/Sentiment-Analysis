@echo off
cd /d "%~dp0"
echo Starting the dashboard. Open http://localhost:8501 in your browser.
echo Keep this window open while you use it. Close it to stop the dashboard.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
