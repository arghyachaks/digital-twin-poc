@echo off
title Train LSTM remaining-life challenger
cd /d "%~dp0.."
set DISABLE_SQLALCHEMY_CEXT_RUNTIME=1
if not exist "ml\models\manifest.json" ( echo Run scripts\6_train_models.bat first. & pause & exit /b 1 )
echo Installing PyTorch (CPU build, first time only, ~200 MB)...
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu
".venv\Scripts\python.exe" ml\train_lstm.py
echo.
echo Comparison: ml\reports\rul_model_comparison.md  (also in MLflow)
pause
