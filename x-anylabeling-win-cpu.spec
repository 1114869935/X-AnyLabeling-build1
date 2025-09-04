# build-windows-final.spec
# 这是包含了针对 Ultralytics/YOLOv8 库特殊修复的最终版本。

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

sys.setrecursionlimit(5000)

# --- 动态定位 ultralytics 库的安装路径 ---
# 无论它被安装在哪里，我们都能准确找到它
import ultralytics
ultralytics_path = Path(ultralytics.__file__).parent


# --- 自动搜集所有核心库可能遗漏的隐藏模块 ---
hidden_imports = []
hidden_imports.extend(collect_submodules('onnxruntime'))
hidden_imports.extend(collect_submodules('skimage'))
hidden_imports.extend(collect_submodules('scipy'))
hidden_imports.extend(collect_submodules('sklearn'))
hidden_imports.extend(collect_submodules('ultralytics')) # 【终极修复】强制包含 ultralytics 所有子模块
hidden_imports.append('pkg_resources.extern')


a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=[],
    # 将项目核心文件夹和【终极修复】所需的数据文件都打包进去
    datas=[
        ('anylabeling/configs', 'anylabeling/configs'),
        ('anylabeling/views', 'anylabeling/views'),
        ('anylabeling/services', 'anylabeling/services'),
        # 【终极修复】强制包含 ultralytics 库的所有数据文件（如 .yaml）
        (str(ultralytics_path / 'cfg'), 'ultralytics/cfg')
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=['runtime_hook.py'],  # 保留路径修复钩子
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='X-AnyLabeling-Final',  # 使用全新的名字，代表最终版
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,  # 【强制】我们必须保留这个窗口，它是成功的唯一凭证
    icon='anylabeling/resources/images/icon.ico',
)
