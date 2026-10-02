@echo off
title CDU-100 Digital Twin
cd /d "%~dp0"
echo ============================================================
echo   CDU-100 Digital Twin  -  FastAPI + 3D twin + AI assistant
echo ============================================================
set "PY=%CD%\.venv\Scripts\python.exe"
if exist "%PY%" goto deps

rem ---- 1. find a base Python (Anaconda / Miniconda / python.org) or install one
set "BASEPY="
for %%P in ("%USERPROFILE%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%LOCALAPPDATA%\Programs\Python\Python312\python.exe") do if exist %%P if not defined BASEPY set "BASEPY=%%~P"
if defined BASEPY goto mkvenv
echo Python not found - installing Python 3.12 with winget...
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
set "BASEPY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
:mkvenv
echo Creating virtual environment (.venv) with %BASEPY%
"%BASEPY%" -m venv .venv
if not exist "%PY%" ( echo [ERROR] could not create .venv & pause & exit /b 1 )

:deps
echo Installing / checking Python packages (first run takes a few minutes)...
"%PY%" -m pip install -q --disable-pip-version-check -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed - see messages above & pause & exit /b 1 )

rem ---- 3D assets for the browser (exported once from Blender)
if not exist "Content_Source\Web\SM_Pump_Centrifugal.glb" call scripts\1_build_assets_blender.bat
if not exist ".env" copy .env.example .env >nul

set "PORT=8000"
for /f "tokens=1,2 delims==" %%A in ('findstr /b "PORT=" .env 2^>nul') do set "PORT=%%B"
echo Starting on http://localhost:%PORT%  (close this window to stop)
start "" /min cmd /c "timeout /t 8 >nul & start http://localhost:%PORT%"
"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT% --log-level warning
pause
