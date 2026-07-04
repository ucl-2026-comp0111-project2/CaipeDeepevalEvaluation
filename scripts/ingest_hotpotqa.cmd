@echo off
setlocal
REM Runs HotpotQA ingestion with the repository default sample settings.
REM Extra CLI options can be appended after the script name; later options override earlier ones.

set "REPO_ROOT=%~dp0.."
set "PYTHON_BIN=python"
if not "%PYTHON%"=="" set "PYTHON_BIN=%PYTHON%"

"%PYTHON_BIN%" "%REPO_ROOT%\src\deepeval_eval\hotpotqa_deepeval.py" ingest --limit 100 --questions-per-category 50 --max-docs 1000 --batch-size 50 %*
exit /b %ERRORLEVEL%
