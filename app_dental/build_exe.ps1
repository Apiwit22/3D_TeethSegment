# =========================================================
# DentalSegApp - PyInstaller build script (PowerShell)
# =========================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
Write-Host "[INFO] Project root: $PWD"

# ---- activate venv if exists ----
$venvActivate = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
  Write-Host "[INFO] Activating venv..."
  . $venvActivate
} else {
  Write-Host "[WARN] venv not found at $venvActivate (using current python)"
}

# ---- sanity checks ----
if (!(Test-Path "app\__main__.py")) { throw "Missing app\__main__.py" }
if (!(Test-Path "registry.yaml"))  { throw "Missing registry.yaml" }
if (!(Test-Path "assets\icons\app.ico")) { Write-Host "[WARN] Missing assets\icons\app.ico" }

# ---- clean old build ----
Write-Host "[INFO] Cleaning old build artifacts..."
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist")  { Remove-Item -Recurse -Force "dist" }
Get-ChildItem -Filter "*.spec" -ErrorAction SilentlyContinue | Remove-Item -Force

# ---- ensure pyinstaller installed ----
try {
  python -c "import PyInstaller" | Out-Null
} catch {
  Write-Host "[INFO] Installing pyinstaller + hooks..."
  pip install -U pyinstaller pyinstaller-hooks-contrib
}

# ---- build ----
Write-Host "[INFO] Building exe (onedir)..."

pyinstaller -y --noconsole --onedir `
  --name DentalSegApp `
  --icon "assets\icons\app.ico" `
  --add-data "registry.yaml;." `
  --add-data "assets;assets" `
  --add-data "models;models" `
  --hidden-import app.models.meshsegnet_repo_wrapper `
  --hidden-import app.models.pointnetpp `
  --hidden-import torch_cluster `
  --hidden-import torch_scatter `
  --hidden-import torch_sparse `
  --hidden-import torch_spline_conv `
  --hidden-import torch_geometric `
  --collect-all torch_cluster `
  --collect-all torch_scatter `
  --collect-all torch_sparse `
  --collect-all torch_spline_conv `
  --collect-all torch_geometric `
  app\__main__.py

Write-Host "[OK] Build finished."
Write-Host "[OK] EXE: $PWD\dist\DentalSegApp\DentalSegApp.exe"