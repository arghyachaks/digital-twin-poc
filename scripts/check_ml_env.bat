@echo off
cd /d "%~dp0.."
if not exist logs mkdir logs
set "OUT=logs\ml_env_check.txt"
echo ML environment check %DATE% %TIME% > "%OUT%"
for %%P in (".venv\Scripts\python.exe" "%USERPROFILE%\anaconda3\python.exe") do (
  echo. >> "%OUT%"
  echo ===== %%~P >> "%OUT%"
  for %%M in (numpy scipy pandas sklearn matplotlib joblib xgboost mlflow torch) do (
    "%%~P" -c "import %%M as m; print('OK   %%M', getattr(m,'__version__','?'))" >> "%OUT%" 2>&1 || echo FAIL %%M >> "%OUT%"
  )
)
type "%OUT%"
echo.
echo Saved to %OUT%
pause
