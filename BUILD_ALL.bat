@echo off
cd /d "%~dp0"
call scripts\1_build_assets_blender.bat || (pause & exit /b 1)
call scripts\2_build_twin_unreal.bat
pause
