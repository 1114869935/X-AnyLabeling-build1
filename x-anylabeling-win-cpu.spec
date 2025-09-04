import sys
from PyInstaller.utils.hooks import collect_submodules

sys.setrecursionlimit(5000)

# --- 自动搜集所有核心库可能遗漏的隐藏模块 ---
hidden_imports = []
hidden_imports.extend(collect_submodules('onnxruntime'))
hidden_imports.extend(collect_submodules('skimage'))
hidden_imports.extend(collect_submodules('scipy'))
hidden_imports.extend(collect_submodules('sklearn'))
hidden_imports.append('pkg_resources.extern')


a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=[],
    # 将项目核心文件夹整个打包进去，确保万无一失
    datas=[
        ('anylabeling/configs', 'anylabeling/configs'),
        ('anylabeling/views', 'anylabeling/views'),
        ('anylabeling/services', 'anylabeling/services')
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=['runtime_hook.py'],  # 用钩子文件彻底解决路径问题
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='X-AnyLabeling-Patched',  # 使用新名字，避免和旧版本混淆
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,  # 【强制】必须为 True，这是我们观察程序内部的唯一窗口
    icon='anylabeling/resources/images/icon.ico', # Windows图标
)
