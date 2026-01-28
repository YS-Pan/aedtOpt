@echo off
set "SCRIPT_NAME=%~n0"

:: Extract base script name (remove "run_" prefix)
set "BASE_NAME=%SCRIPT_NAME:run_=%"

:: Initialize parameter
set "SKIP_ROWS=0"

:: Check if there's a number after the base script name
for /f "tokens=1,2 delims=_" %%a in ("%BASE_NAME%") do (
    set "SCRIPT_BASE=%%a"
    if not "%%b"=="" set "SKIP_ROWS=%%b"
)

:: If SCRIPT_BASE is set, update BASE_NAME
if defined SCRIPT_BASE (
    set "BASE_NAME=%SCRIPT_BASE%"
)

:: Run Python script with parameter
python "%BASE_NAME%.py" --skip-rows=%SKIP_ROWS%
pause
