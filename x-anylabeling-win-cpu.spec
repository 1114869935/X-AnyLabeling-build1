# -*- mode: python -*-
# vim: ft=python

import sys

sys.setrecursionlimit(5000)  # required on Windows

a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=[],
    datas=[
        ('anylabeling/configs/auto_labeling/*.yaml', 'anylabeling/configs/auto_labeling'),
        ('anylabeling/configs/*.yaml', 'anylabeling/configs'),
        ('anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui', 'anylabeling/views/labeling/widgets/auto_labeling'),
        ('anylabeling/services/auto_labeling/configs/bert/*', 'anylabeling/services/auto_labeling/configs/bert'),
        ('anylabeling/services/auto_labeling/configs/clip/*', 'anylabeling/services/auto_labeling/configs/clip'),
        ('anylabeling/services/auto_labeling/configs/ppocr/*', 'anylabeling/services/auto_labeling/configs/ppocr'),
        ('anylabeling/services/auto_labeling/configs/ram/*', 'anylabeling/services/auto_labeling/configs/ram')
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=['runtime_hook.py'],  # <---【修改一】在这里加上了钩子文件
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='X-AnyLabeling-CPU-Debug',  # 我加了-Debug后缀，方便你识别
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,  # <---【修改二】这里改成了 True，强制显示调试窗口
    icon='anylabeling/resources/images/icon.ico', # 建议Windows用.ico图标，但不是强制
)

# 下面这部分在Windows构建时会被自动忽略，保留也没关系
app = BUNDLE(
    exe,
    name='X-AnyLabeling.app',
    icon='anylabeling/resources/images/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
