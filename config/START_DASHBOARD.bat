@echo off
REM Quick Start Script for COLORFUL ANIMATED Visual Dashboard
REM Double-click this file to launch the dashboard

echo ========================================
echo 🚀 5G FL-QoS ANIMATED DASHBOARD
echo ========================================
echo.
echo Starting colorful animated dashboard...
echo Dashboard will open at: http://localhost:8503
echo.
echo Features:
echo - Auto-playing traffic animation
echo - Colorful gradient UI
echo - Real-time metrics
echo - 3D network view
echo.
echo Press Ctrl+C to stop the dashboard
echo ========================================
echo.

cd /d "%~dp0"
streamlit run visual_dashboard_v2.py --server.port=8503

pause
