@echo off
setlocal
cd /d "%~dp0.."
call scripts\find_tools.bat
if not defined BLENDER_EXE ( echo [ERROR] Blender not found. set BLENDER_EXE=C:\path\to\blender.exe & exit /b 1 )
if not exist logs mkdir logs
echo Building 3D assets + 2K textures in Blender (headless)...
"%BLENDER_EXE%" -b --factory-startup -P "%CD%\blender\build_assets.py" > logs\blender_build.log 2>&1
if errorlevel 1 ( echo [ERROR] Blender build failed - see logs\blender_build.log & type logs\blender_build.log | findstr /i "error traceback" & exit /b 1 )
findstr /c:"[TwinBuilder]" logs\blender_build.log
echo OK - meshes in Content_Source\Meshes
