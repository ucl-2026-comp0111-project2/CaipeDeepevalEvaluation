@echo off
setlocal
REM Runs HotpotQA evaluation with the repository default evaluation settings.
REM Extra CLI options can be appended after the script name; later options override earlier ones.

set "REPO_ROOT=%~dp0.."
set "PYTHON_BIN=python"
if not "%PYTHON%"=="" set "PYTHON_BIN=%PYTHON%"

"%PYTHON_BIN%" "%REPO_ROOT%\src\deepeval_eval\hotpotqa_deepeval.py" eval --max-items 10 --top-k 5 --max-context-chars 12000 %*
exit /b %ERRORLEVEL%
