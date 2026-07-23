@echo off
setlocal

REM Run EnterpriseRAG-Bench agentic evaluation via CAIPE supervisor A2A endpoint.
REM Additional CLI arguments can be passed to override the defaults.

set "REPO_ROOT=%~dp0.."
set "PYTHON_BIN=python"
if not "%PYTHON%"=="" set "PYTHON_BIN=%PYTHON%"

if "%CAIPE_SUPERVISOR_URL%"=="" set "CAIPE_SUPERVISOR_URL=http://localhost:8000"

"%PYTHON_BIN%" "%REPO_ROOT%\src\deepeval_eval\deepeval_evaluator.py" eval ^
  --dataset-name enterprise ^
  --agentic ^
  --supervisor-url "%CAIPE_SUPERVISOR_URL%" ^
  --max-items 10 ^
  --top-k 3 ^
  --max-context-chars 6000 ^
  %*

exit /b %ERRORLEVEL%