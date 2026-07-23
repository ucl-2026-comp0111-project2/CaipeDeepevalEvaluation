@echo off
setlocal

REM Run EnterpriseRAG-Bench ingestion using the repository defaults.
REM Additional CLI arguments can be passed to override the defaults.

set "REPO_ROOT=%~dp0.."
set "PYTHON_BIN=python"
if not "%PYTHON%"=="" set "PYTHON_BIN=%PYTHON%"

"%PYTHON_BIN%" "%REPO_ROOT%\src\deepeval_eval\ingest.py" ^
  --dataset-name enterprise ^
  --sources confluence jira github hubspot fireflies linear google_drive gmail slack ^
  --limit-per-source 1000 ^
  --num-questions 10 ^
  --questions-per-category 3 ^
  --batch-size 50 ^
  %*

exit /b %ERRORLEVEL%
