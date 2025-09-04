# 这是一个基于您原始配置的、只添加了必要修复的纯净版本。

import sys

sys.setrecursionlimit(5000)

a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=[],
    datas=[
        ('anylabeling/configs', 'anylabeling/configs'),
        ('anylabeling/views', 'anylabeling/views'),
        ('anylabeling/services', 'anylabeling/services')
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=['runtime_hook.py'],  # 【核心修复】只加入这一行，解决路径问题
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='X-AnyLabeling-Fixed',  # 使用新名字，代表“已修复”
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,  # 【安全保障】我们必须保留这个，万一还有问题，它会告诉我们原因
    icon='anylabeling/resources/images/icon.ico',
)
