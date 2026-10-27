@echo off
cd /d "%~dp0"
echo Step 1/3: setting up the project environment (needs internet, takes several minutes)...
python scripts\setup_env.py
if errorlevel 1 goto fail
echo.
echo Step 2/3: running the automated tests...
.venv\Scripts\python.exe scripts\run_tests.py
echo.
echo Step 3/3: smoke check...
.venv\Scripts\python.exe scripts\smoke_check.py
echo.
echo FINISHED. Reports: results\test_report.txt and results\smoke_check.txt. Tell Claude it is done.
pause
exit /b 0
:fail
echo.
echo SETUP FAILED. Read the messages above or logs\setup_env.log, and tell Claude.
pause
exit /b 1
