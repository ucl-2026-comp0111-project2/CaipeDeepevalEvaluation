@echo off
setlocal

REM Run HotpotQA agentic evaluation via CAIPE supervisor A2A endpoint.

set "REPO_ROOT=%~dp0.."
set "PYTHON_BIN=python"
if not "%PYTHON%"=="" set "PYTHON_BIN=%PYTHON%"

if "%CAIPE_SUPERVISOR_URL%"=="" set "CAIPE_SUPERVISOR_URL=http://localhost:8000"

"%PYTHON_BIN%" "%REPO_ROOT%\src\deepeval_eval\hotpotqa_deepeval.py" eval ^
  --agentic ^
  --supervisor-url "%CAIPE_SUPERVISOR_URL%" ^
  --max-items 10 ^
  --top-k 5 ^
  --max-context-chars 12000 ^
  %*

exit /b %ERRORLEVEL%
