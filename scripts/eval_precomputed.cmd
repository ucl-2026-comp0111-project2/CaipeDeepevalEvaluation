@echo off
setlocal

REM Run DeepEval against benchmark ground-truth contexts/reference answers.
REM Additional CLI arguments can be passed to override the defaults.

set "REPO_ROOT=%~dp0.."
set "PYTHON_BIN=python"
if not "%PYTHON%"=="" set "PYTHON_BIN=%PYTHON%"

"%PYTHON_BIN%" "%REPO_ROOT%\src\deepeval_eval\precomputed_deepeval.py" ^
  --benchmark hotpotqa ^
  --max-items 20 ^
  --answer-mode reference ^
  %*

exit /b %ERRORLEVEL%
