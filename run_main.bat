@echo off
set "SCRIPT_NAME=%~n0"
set "PYTHON_FILE=%SCRIPT_NAME:run_=%.py"
python "%PYTHON_FILE%"
pause