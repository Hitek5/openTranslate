@echo off
setlocal

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js not found. Install Node.js 18+ from https://nodejs.org/
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm not found. Reinstall Node.js with npm.
  exit /b 1
)

echo Installing dependencies...
call npm install
if errorlevel 1 (
  echo [ERROR] npm install failed.
  exit /b 1
)

echo Starting dev server...
call npm run dev -- --host 0.0.0.0 --port 5173

endlocal
