import sys

sys.setrecursionlimit(5000)

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
    # 【核心修复】强制包含 onnxruntime 和其他库可能遗漏的隐藏依赖
    hiddenimports=[
        'onnxruntime.capi._pybind_state',
        'scipy.special._cdflib',
        'scipy._lib.messagestream',
        'sklearn.utils._cython_blas',
        'skimage.io._plugins',
        'pkg_resources.extern'
    ],
    hookspath=[],
    runtime_hooks=['runtime_hook.py'],  # 永久修复路径问题
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='X-AnyLabeling-CPU-Patched',  # 新名字，表示已修复
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,  # 保留诊断能力，这是程序的“黑匣子”
    icon='anylabeling/resources/images/icon.ico',
)

# macOS 部分在 Windows 构建时会被忽略
app = BUNDLE(
    exe,
    name='X-AnyLabeling.app',
    icon='anylabeling/resources/images/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
