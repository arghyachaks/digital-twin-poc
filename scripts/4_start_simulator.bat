@echo off
cd /d "%~dp0.."
where python >nul 2>nul || ( echo [ERROR] Python not on PATH - open an Anaconda Prompt and run: python backend\simulator.py & pause & exit /b 1 )
python -m pip install -q -r backend\requirements.txt
python backend\simulator.py %*
pause
