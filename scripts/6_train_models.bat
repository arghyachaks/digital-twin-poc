@echo off
title Train pump condition models
cd /d "%~dp0.."
set DISABLE_SQLALCHEMY_CEXT_RUNTIME=1
if not exist ".venv\Scripts\python.exe" ( echo Run START.bat once first to create the Python environment. & pause & exit /b 1 )
echo Installing ML packages (first time: a few minutes)...
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r ml\requirements.txt
if not exist "ml\data\pump_runs.csv.gz" ".venv\Scripts\python.exe" ml\generate_dataset.py
".venv\Scripts\python.exe" ml\train.py
if errorlevel 1 ( echo. & echo TRAINING FAILED - see the error above. & pause & exit /b 1 )
echo.
echo Models saved to ml\models  -  report: ml\reports\model_report.md
echo Restart START.bat so the twin loads the new models. Browse runs with scripts\7_mlflow_ui.bat
pause
