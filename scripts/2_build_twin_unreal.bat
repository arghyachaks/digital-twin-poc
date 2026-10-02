@echo off
setlocal
cd /d "%~dp0.."
call scripts\find_tools.bat
if not defined UE_DIR ( echo [ERROR] Unreal Engine 5 not found. set UE_DIR=C:\Program Files\Epic Games\UE_5.x & exit /b 1 )
set "PROJ=%CD%\unreal\RefineryTwin\RefineryTwin.uproject"
set "UE_EXE=%UE_DIR%\Engine\Binaries\Win64\UnrealEditor.exe"
powershell -NoProfile -Command "(Get-Content 'unreal\RefineryTwin\RefineryTwin.uproject.template') -replace '__ENGINE__','%UE_VER%' | Set-Content -Encoding UTF8 '%PROJ%'"
tasklist /fi "imagename eq UnrealEditor.exe" | find /i "UnrealEditor.exe" >nul && ( echo [ERROR] Close all Unreal Editor windows first, then re-run. & pause & exit /b 1 )
rem Clean rebuild: everything under /Game/DigitalTwin is generated from Content_Source
if exist "unreal\RefineryTwin\Content\DigitalTwin" rmdir /s /q "unreal\RefineryTwin\Content\DigitalTwin"
if not exist "unreal\RefineryTwin\Content" mkdir "unreal\RefineryTwin\Content"
set "PY=%CD%\unreal\build_twin.py"
set "PY=%PY:\=/%"
echo [1/2] Building the twin in Unreal %UE_VER% (first run compiles shaders - be patient)...
"%UE_EXE%" "%PROJ%" /Engine/Maps/Entry -ExecutePythonScript="%PY%" -log=RefineryTwinBuild.log -unattended -nosplash
findstr /c:"[TwinBuilder]" "unreal\RefineryTwin\Saved\Logs\RefineryTwinBuild.log"
echo [2/2] Opening the Refinery Twin level...
start "" "%UE_EXE%" "%PROJ%" /Game/DigitalTwin/Maps/L_Refinery
