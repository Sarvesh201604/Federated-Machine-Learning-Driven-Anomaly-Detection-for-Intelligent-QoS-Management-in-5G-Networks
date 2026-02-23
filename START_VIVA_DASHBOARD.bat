@echo off
REM Script to launch the Educational Step-by-Step Viva Dashboard
REM This is perfect for review panels who want to understand the MATH and ML.

echo ========================================
echo 🎓 5G FL-QoS VIVA PRESENTATION DASHBOARD
echo ========================================
echo.
echo Starting the Step-by-Step Proof Dashboard...
echo Dashboard will open natively in your browser.
echo.
echo Features:
echo - Raw Traffic Injection
echo - Live Neural Network Probability
echo - Full Mathematical Proof of the Routing Algorithm
echo.
echo Press Ctrl+C to stop the dashboard
echo ========================================
echo.

cd /d "%~dp0"
streamlit run viva_presentation_dashboard.py --server.port=8504

pause
