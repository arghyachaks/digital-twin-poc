@echo off
cd /d "%~dp0.."
call scripts\find_tools.bat
start "" "%UE_DIR%\Engine\Binaries\Win64\UnrealEditor.exe" "%CD%\unreal\RefineryTwin\RefineryTwin.uproject" /Game/DigitalTwin/Maps/L_Refinery
