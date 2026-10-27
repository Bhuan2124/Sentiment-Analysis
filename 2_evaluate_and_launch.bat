@echo off
cd /d "%~dp0"
if not exist models\distilbert\config.json (
  echo DistilBERT training has not finished yet: models\distilbert is missing. Run this after training ends.
  pause
  exit /b 1
)
echo Final scoring on the test set...
.venv\Scripts\python.exe scripts\evaluate_models.py
echo.
echo Starting the dashboard at http://localhost:8501 (close this window to stop it)...
.venv\Scripts\python.exe -m streamlit run app.py
pause
