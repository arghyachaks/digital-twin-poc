@echo off
title Generate pump training dataset
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" ( echo Run START.bat once first to create the Python environment. & pause & exit /b 1 )
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r ml\requirements.txt
".venv\Scripts\python.exe" ml\generate_dataset.py %*
echo.
echo Dataset in ml\data  (see ml\data\dataset_card.md and ml\data\plots)
pause
