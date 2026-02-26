@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [INFO] Project root: %CD%

REM ---- activate venv if exists ----
if exist "venv\Scripts\activate.bat" (
  call "venv\Scripts\activate.bat"
) else (
  echo [WARN] venv not found (using current python)
)

REM ---- check scipy ----
python -c "import scipy; print('scipy OK')" >nul 2>&1
if errorlevel 1 (
  echo [INFO] scipy missing -> installing...
  pip install scipy
)

REM ---- ensure pyinstaller installed ----
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
  pip install -U pyinstaller pyinstaller-hooks-contrib
)

REM ---- clean ----
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
for %%F in (*.spec) do del /q "%%F"

echo [INFO] Build onedir exe...

pyinstaller -y --noconsole --onedir ^
  --name DentalSegApp ^
  --icon "assets\icons\app.ico" ^
  --add-data "registry.yaml;." ^
  --add-data "assets;assets" ^
  --add-data "models;models" ^
  --hidden-import scipy ^
  --collect-all scipy ^
  --hidden-import app.models.meshsegnet_repo_wrapper ^
  --hidden-import app.models.pointnetpp ^
  --hidden-import torch_cluster ^
  --hidden-import torch_scatter ^
  --hidden-import torch_sparse ^
  --hidden-import torch_spline_conv ^
  --hidden-import torch_geometric ^
  --collect-all torch_cluster ^
  --collect-all torch_scatter ^
  --collect-all torch_sparse ^
  --collect-all torch_spline_conv ^
  --collect-all torch_geometric ^
  app\__main__.py

if errorlevel 1 (
  echo [ERROR] Build failed.
  exit /b 1
)

echo [OK] EXE: %CD%\dist\DentalSegApp\DentalSegApp.exe
pause