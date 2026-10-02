@echo off
rem Locates Blender and Unreal Engine 5. Override by setting BLENDER_EXE / UE_DIR before calling.
if not defined BLENDER_EXE (
  for /d %%D in ("C:\Program Files\Blender Foundation\Blender*") do if exist "%%D\blender.exe" set "BLENDER_EXE=%%D\blender.exe"
)
if not defined BLENDER_EXE (
  for /d %%D in ("C:\Program Files (x86)\Steam\steamapps\common\Blender") do if exist "%%D\blender.exe" set "BLENDER_EXE=%%D\blender.exe"
)
if not defined UE_DIR (
  for /d %%D in ("C:\Program Files\Epic Games\UE_5.*") do if exist "%%D\Engine\Binaries\Win64\UnrealEditor.exe" set "UE_DIR=%%D"
)
if not defined UE_DIR (
  for /d %%D in ("D:\Epic Games\UE_5.*" "D:\Program Files\Epic Games\UE_5.*") do if exist "%%D\Engine\Binaries\Win64\UnrealEditor.exe" set "UE_DIR=%%D"
)
if defined UE_DIR for %%I in ("%UE_DIR%") do set "UE_NAME=%%~nxI"
if defined UE_NAME set "UE_VER=%UE_NAME:UE_=%"
echo Blender : %BLENDER_EXE%
echo Unreal  : %UE_DIR%  (version %UE_VER%)
