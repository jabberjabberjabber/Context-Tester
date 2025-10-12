@echo off
REM Launcher for Context Tester GUI
REM Uses uv if available, otherwise falls back to python

where uv >nul 2>nul
if %ERRORLEVEL% == 0 (
    echo Starting GUI with uv...
    uv run plot_gui.py
) else (
    echo Starting GUI with python...
    python plot_gui.py
)

pause
