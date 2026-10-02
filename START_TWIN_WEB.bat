@echo off
title CDU-100 Digital Twin - simulator + web viewer
cd /d "%~dp0"
rem ---- 1. find Python (PATH, Anaconda, Miniconda) or install it with winget
set "PY="
for %%P in ("%USERPROFILE%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%LOCALAPPDATA%\Programs\Python\Python312\python.exe") do if not defined PY if exist %%P set "PY=%%~P"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY echo %%P | find /i "WindowsApps" >nul || set "PY=%%P"
if not defined PY (
  echo Python not found - installing Python 3.12 with winget...
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
echo Using Python: %PY%
rem ---- 2. dependencies
"%PY%" -m pip install -q -r backend\requirements.txt
rem ---- 3. web 3D assets (GLB) from Blender, if not exported yet
if not exist "Content_Source\Web\SM_Pump_Centrifugal.glb" (
  echo Exporting 3D assets for the web viewer with Blender...
  call scripts\1_build_assets_blender.bat
)
rem ---- 4. simulator + web server (opens your browser)
"%PY%" backend\simulator.py %*
pause
