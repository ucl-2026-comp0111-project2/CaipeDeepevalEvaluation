@echo off
setlocal
REM Runs EnterpriseRAG-Bench ingestion with the repository default sample settings.
REM Extra CLI options can be appended after the script name; later options override earlier ones.

set "REPO_ROOT=%~dp0.."
set "PYTHON_BIN=python"
if not "%PYTHON%"=="" set "PYTHON_BIN=%PYTHON%"

"%PYTHON_BIN%" "%REPO_ROOT%\src\deepeval_eval\enterprise_deepeval.py" ingest --sources confluence jira github hubspot fireflies linear google_drive gmail slack --limit-per-source 1000 --num-questions 10 --questions-per-category 3 --batch-size 50 %*
exit /b %ERRORLEVEL%
