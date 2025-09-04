# 文件路径: anylabeling/app.py
# 【最终·强制坐标系修复版 - 完整文件】

import os
import sys

# ======================= 【核心修复点】 =======================
# 在导入任何 PyQt/Qt 组件之前，设置所有最高优先级的环境变量
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = ""
if sys.platform == "win32":
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
# =============================================================

# Temporary fix for: bus error
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Suppress ICC profile warnings
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.gui.icc=false"

import argparse
import codecs
import logging

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
from PyQt5 import QtCore, QtWidgets

from anylabeling.app_info import __appname__, __version__, __url__
from anylabeling.config import get_config
from anylabeling import config as anylabeling_config
from anylabeling.views.mainwindow import MainWindow
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils import new_icon, gradient_text
from anylabeling.views.labeling.utils.update_checker import (
    check_for_updates_async,
)

# NOTE: Do not remove this import, it is required for loading translations
from anylabeling.resources import resources


def main():
    parser = argparse.ArgumentParser()
    # ... (所有 argparse 的代码保持不变) ...
    parser.add_argument(
        "--epsilon",
        type=float,
        help="epsilon to find nearest vertex on canvas",
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # ... (所有 args 的处理代码保持不变) ...
    if hasattr(args, "label_flags"):
        if os.path.isfile(args.label_flags):
            with codecs.open(args.label_flags, "r", encoding="utf-8") as f:
                args.label_flags = yaml.safe_load(f)
        else:
            args.label_flags = yaml.safe_load(args.label_flags)

    config_from_args = args.__dict__
    reset_config = config_from_args.pop("reset_config")
    filename = config_from_args.pop("filename")
    output = config_from_args.pop("output")
    config_file_or_yaml = config_from_args.pop("config")
    logger_level = config_from_args.pop("logger_level")
    no_auto_update_check = config_from_args.pop("no_auto_update_check", False)

    logger.setLevel(getattr(logging, logger_level.upper()))
    logger.info(
        f"🚀 {gradient_text(f'X-AnyLabeling v{__version__} launched!')}"
    )
    logger.info(f"⭐ If you like it, give us a star: {__url__}")
    anylabeling_config.current_config_file = config_file_or_yaml
    config = get_config(config_file_or_yaml, config_from_args, show_msg=True)

    if not config["labels"] and config["validate_label"]:
        logger.error(
            "--labels must be specified with --validatelabel or "
            "validate_label: exact in the config file "
            "(ex. ~/.xanylabelingrc)."
        )
        sys.exit(1)

    output_file = None
    output_dir = None
    if output is not None:
        if output.endswith(".json"):
            output_file = output
        else:
            output_dir = output

    language = config.get("language", QtCore.QLocale.system().name())
    translator = QtCore.QTranslator()
    loaded_language = translator.load(
        ":/languages/translations/" + language + ".qm"
    )
    
    # 再次调用 setAttribute，与环境变量配合，形成最强保障
    if sys.platform == "win32":
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

    app = QtWidgets.QApplication(sys.argv)
    app.processEvents()

    app.setApplicationName(__appname__)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(new_icon("icon"))
    if loaded_language:
        app.installTranslator(translator)
    else:
        logger.warning(
            f"Failed to load translation for {language}. "
            "Using default language.",
        )
    win = MainWindow(
        app,
        config=config,
        filename=filename,
        output_file=output_file,
        output_dir=output_dir,
    )

    if reset_config:
        logger.info(f"Resetting Qt config: {win.settings.fileName()}")
        win.settings.clear()
        sys.exit(0)

    if not no_auto_update_check:

        def delayed_update_check():
            check_for_updates_async(timeout=5)

        QtCore.QTimer.singleShot(2000, delayed_update_check)

    win.showMaximized()
    win.raise_()
    sys.exit(app.exec())


# this main block is required to generate executable by pyinstaller
if __name__ == "__main__":
    main()
