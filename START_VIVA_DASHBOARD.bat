@echo off
REM =============================================================================
REM VIVA PRESENTATION DASHBOARD LAUNCHER
REM =============================================================================
REM Purpose: Launches interactive demonstration for reviewers
REM Shows: Real-time ML predictions + Animated routing decisions
REM Port: http://localhost:8504
REM =============================================================================

color 0A
echo.
echo ================================================================================
echo    🎓 VIVA PRESENTATION DASHBOARD - FOR REVIEWERS
echo ================================================================================
echo.
echo    Purpose: Interactive demonstration of Federated Learning-based
echo             Anomaly Detection for 5G QoS Management
echo.
echo    What Reviewers Will See:
echo    ✅ Live traffic generation (Normal, Suspicious, Malicious)
echo    ✅ Real-time ML anomaly predictions
echo    ✅ Mathematical proof of routing cost calculation
echo    ✅ Animated network showing intelligent routing decisions
echo.
echo ================================================================================
echo    🚀 Starting Dashboard...
echo ================================================================================
echo.
echo    [1/3] Initializing system...

cd /d "%~dp0"

echo    [2/3] Training Federated Learning model...
echo    [3/3] Launching web interface...
echo.
echo ================================================================================
echo    ✅ Dashboard is starting!
echo ================================================================================
echo.
echo    📱 The dashboard will open in your browser automatically
echo    🌐 URL: http://localhost:8504
echo.
echo    💡 TIP: Read the instructions at the top of the dashboard for best results
echo.
echo    ⏹️  To stop: Press Ctrl+C in this window
echo.
echo ================================================================================
echo.

streamlit run viva_presentation_dashboard.py --server.port=8504 --server.headless=true

echo.
echo ================================================================================
echo    Dashboard stopped. You can close this window.
echo ================================================================================
pause
