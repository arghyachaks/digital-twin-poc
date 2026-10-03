@echo off
title MLflow - CDU-100 model tracking
cd /d "%~dp0.."
set DISABLE_SQLALCHEMY_CEXT_RUNTIME=1
start "" cmd /c "timeout /t 6 >nul & start http://localhost:5000"
".venv\Scripts\python.exe" -m mlflow ui --backend-store-uri sqlite:///ml/mlflow.db --port 5000
pause
