@echo off
title Push CDU-100 Digital Twin to GitHub (develop)
cd /d "%~dp0"
set "REPO=https://github.com/arghyachaks/digital-twin-poc.git"
set "BRANCH=develop"
where git >nul 2>nul || ( echo [ERROR] Git is not installed. Get it from https://git-scm.com/download/win & pause & exit /b 1 )

if not exist ".git" git init -q
git config user.name  >nul 2>nul || git config user.name  "Arghya Chakraborty"
git config user.email >nul 2>nul || git config user.email "arghyachaks@gmail.com"
git remote remove origin >nul 2>nul
git remote add origin %REPO%

echo Fetching %REPO% ...
git fetch origin
rem Build on top of what is already on GitHub (no force push): develop if it exists, else main/master
set "BASE="
git rev-parse --verify -q origin/%BRANCH% >nul && set "BASE=origin/%BRANCH%"
if not defined BASE git rev-parse --verify -q origin/main >nul && set "BASE=origin/main"
if not defined BASE git rev-parse --verify -q origin/master >nul && set "BASE=origin/master"
git checkout -q -B %BRANCH%
if defined BASE (
  echo Basing on %BASE%
  git reset -q --mixed %BASE%
)

git add -A
git commit -q -F scripts\commit_message.txt || echo (nothing new to commit)
echo Pushing to %BRANCH% ...
git push -u origin %BRANCH%
if errorlevel 1 ( echo [ERROR] Push failed - see the message above. & pause & exit /b 1 )
echo.
echo Done: https://github.com/arghyachaks/digital-twin-poc/tree/%BRANCH%
pause
