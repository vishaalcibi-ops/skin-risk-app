@echo off
echo Setting up Skin Disease Identifier...
echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies. Please check your internet connection and python installation.
    pause
    exit /b
)
echo Starting the application...
python app.py
pause
