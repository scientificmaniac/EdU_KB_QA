# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置。

打包命令::

    pyinstaller 智能问答系统.spec --noconfirm

产物::

    dist/智能问答系统.exe

数据文件（示例知识库、配置模板）会内嵌进 exe；首次运行时由
``career_qa.cli.bootstrap_bundled_data`` 释放到 exe 同级目录，
这样用户可以直接编辑知识库，而不必面对只读的临时解包目录。
"""

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('knowledge_base.json', '.'),
        ('config.example.json', '.'),
    ],
    # career_qa.gui 虽由 cli.main() 显式导入、可被静态分析捕获，
    # 但 app.py 通过 __getattr__ 延迟导出 ChatGUI 属于运行期分支，
    # 这里显式声明以策万全，避免打包后找不到界面模块。
    hiddenimports=['requests', 'career_qa.gui'],
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
    a.binaries,
    a.datas,
    [],
    name='智能问答系统',
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
)
