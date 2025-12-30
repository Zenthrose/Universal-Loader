# -*- mode: python ; coding: utf-8 -*-
"""
Universal-Loader GUI Specification for PyInstaller
================================================
Build standalone Windows executable:
    pip install pyinstaller
    pyinstaller universal_loader.spec
"""

import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

a = Analysis(
    ['universal_loader_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('build/vulkangguf.cp312-mingw_x86_64_msvcrt_gnu.pyd', '.'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.scrolledtext',
        'threading',
        'importlib',
        'importlib.util',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='universal_loader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
