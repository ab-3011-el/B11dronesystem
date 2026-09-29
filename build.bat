@echo off
REM ---------------------------------------------------------------
REM  Build DroneFlightSim.exe (standalone, no Python needed to run)
REM  Usage:  build.bat          -> windowed build
REM          build.bat debug    -> keeps a console window for errors
REM ---------------------------------------------------------------
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH. Install Python 3.10-3.12 first.
    exit /b 1
)

set MODE=--windowed
if /I "%1"=="debug" set MODE=--console

echo [1/3] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [2/3] Building executable...
python -m PyInstaller --noconfirm --clean --onefile %MODE% --name DroneFlightSim ^
    --collect-all ursina --collect-all panda3d ^
    --collect-all panda3d_gltf --collect-all panda3d_simplepbr ^
    "main (1).py"
if errorlevel 1 (
    echo [ERROR] PyInstaller failed.
    exit /b 1
)

echo [3/3] Done.  Executable: %~dp0dist\DroneFlightSim.exe
endlocal
