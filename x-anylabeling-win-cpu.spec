# build-windows-final.spec
# 这是最终的、最完整的构建配置文件。

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
```4.  保存并退出 (`Ctrl+X` -> `Y` -> `回车`)。

#### **第二步：创建 `runtime_hook.py` 文件**

这个小文件至关重要，它能确保 `.exe` 程序总能找到旁边的 `models` 文件夹。

1.  如果您已经创建过此文件，可以跳过。否则，请创建它：
    ```bash
    nano runtime_hook.py
    ```
2.  粘贴以下代码：
    ```python
    import os
    import sys

    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    ```
3.  保存并退出 (`Ctrl+X` -> `Y` -> `回车`)。

#### **第三步：替换您的 GitHub Actions 工作流文件**

这是最后一块拼图。我们将告诉 GitHub Actions 使用我们刚刚创建的那个全新的、完美的 `.spec` 文件。

1.  用 `nano` 打开您的 `.github/workflows/your-workflow-name.yml` 文件。
2.  **删除里面的所有内容**。
3.  **将下面的工作流代码完整地复制并粘贴进去。**

```yaml
name: Build Patched Windows App

on:
  workflow_dispatch:
  push:
    branches:
      - main

permissions:
  contents: write

jobs:
  build:
    runs-on: windows-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          lfs: true

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: 'pip'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pyinstaller

      - name: Build EXE with Final Spec
        # 这是最关键的一步：我们强制它使用全新的最终配置文件
        run: pyinstaller --noconfirm build-windows-final.spec

      - name: Copy Models Folder
        # 构建产物现在位于一个名为 'X-AnyLabeling-Patched' 的文件夹里
        run: Copy-Item -Path models -Destination dist/X-AnyLabeling-Patched -Recurse -Force

      - name: Package to Zip
        run: |
          Compress-Archive -Path dist/X-AnyLabeling-Patched/* -DestinationPath X-AnyLabeling-Win-Patched-v${{ github.run_number }}.zip
          echo "ZIP_NAME=X-AnyLabeling-Win-Patched-v${{ github.run_number }}.zip" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append

      - name: Upload and Release
        uses: softprops/action-gh-release@v2
        with:
          tag_name: v${{ github.run_number }}-patched
          name: "Windows Patched Build #${{ github.run_number }}"
          body: "这是一个修复版构建，旨在解决模型加载问题。"
          files: ${{ env.ZIP_NAME }}
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
