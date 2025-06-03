@echo off
REM Start Flask server
start cmd /k python Crm_ml.py
REM Wait a few seconds for server to start
timeout /t 3
REM Open the report in browser
start http://127.0.0.1:5000/report