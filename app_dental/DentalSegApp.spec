# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('registry.yaml', '.'), ('assets', 'assets'), ('models', 'models')]
binaries = []
hiddenimports = ['app.models.meshsegnet_repo_wrapper', 'app.models.meshsegnet_runner', 'app.models.meshsegnet', 'app.models.pointnetpp_runner', 'app.models.pointnetpp', 'app.models.tsmdl_runner', 'app.models.imeshsegnet_knn2', 'scipy', 'scipy.spatial', 'scipy.spatial.transform']
hiddenimports += collect_submodules('app.models')
tmp_ret = collect_all('torch_cluster')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch_scatter')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch_sparse')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch_geometric')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app\\__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DentalSegApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\icons\\app.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DentalSegApp',
)
