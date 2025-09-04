# 文件路径: anylabeling/services/auto_labeling/bacteria_autoseg.py
# 【多线程·最终版】 - 将模型推理放到工作线程，防止 UI 卡死

import logging
import traceback
import numpy as np
import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QThread, pyqtSignal

# ==================== 代码修改区域 开始 ====================
from typing import Union

# 我们需要一个通用的 get_resource_path 函数，如果它不在这个文件里，
# 就需要从一个公共的 utils 文件导入，或者直接在这里定义。
# 为确保独立性，我们在此重新定义。
import sys
import os

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
# ==========================================================

from .model import Model
from .bacteria_onnx import BacteriaONNX
from .types import AutoLabelingResult
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

# ==================== 新增工作线程类 开始 ====================
# 1. 创建一个 Worker 对象，它将负责所有繁重的计算
class SegmentationWorker(QObject):
    # 定义信号，用于在工作完成后通知主线程
    finished = pyqtSignal(list)  # 发送处理好的 shapes 列表
    error = pyqtSignal(str)      # 如果出错，发送错误信息

    def __init__(self, model, image, filename):
        super().__init__()
        self.model = model
        self.image = image
        self.filename = filename

    # 这个 run 方法将在后台线程中执行
    @QtCore.pyqtSlot()
    def run(self):
        try:
            logging.info("Worker thread starting segmentation...")
            cv_image = qt_img_to_rgb_cv_img(self.image, self.filename)
            all_masks = self.model.predict_masks(cv_image)

            if all_masks.size == 0:
                logging.info("Worker: No masks found.")
                self.finished.emit([])
                return

            all_shapes = []
            for i in range(all_masks.shape[0]):
                # 后处理依然在工作线程中完成
                mask = all_masks[i]
                if mask.dtype != np.uint8: mask = mask.astype(np.uint8)
                kernel = np.ones((7, 7), np.uint8)
                smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    if cv2.contourArea(contour) < 10: continue
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.004 * perimeter
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    points = approx_contour.reshape(-1, 2).tolist()
                    if len(points) < 3: continue
                    shape = Shape(label="bacteria", shape_type="polygon")
                    for point in points:
                        shape.add_point(QtCore.QPointF(int(point[0]), int(point[1])))
                    all_shapes.append(shape)

            logging.info(f"Worker finished. Found {len(all_shapes)} shapes.")
            self.finished.emit(all_shapes) # 发送结果
        except Exception as e:
            logging.error(f"Error in worker thread: {e}")
            self.error.emit(str(e))
# ==================== 新增工作线程类 结束 ====================


class BacteriaAutoseg(Model):
    # ==================== 新增信号 开始 ====================
    # 这个信号将触发 UI 更新，必须在主类中定义
    _prediction_done = pyqtSignal(AutoLabelingResult)
    # ==================== 新增信号 结束 ====================

    class Meta:
        required_config_names = [ "type", "name", "display_name", "encoder_model_path", "decoder_model_path" ]
        widgets = []
        output_modes = {"polygon": "Polygon"}
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        super().__init__(config_path, on_message)
        
        encoder_rel_path = self.config.get("encoder_model_path")
        decoder_rel_path = self.config.get("decoder_model_path")
        input_size = self.config.get("input_size", 1024)

        encoder_model_abs_path = get_resource_path(encoder_rel_path)
        decoder_model_abs_path = get_resource_path(decoder_rel_path)

        self.model = BacteriaONNX(encoder_path=encoder_model_abs_path, decoder_path=decoder_model_abs_path, input_size=input_size)
        
        # ==================== 初始化线程 开始 ====================
        self.thread = None
        self.worker = None
        # 连接我们自己的信号到 X-AnyLabeling 的 on_message 回调函数
        self._prediction_done.connect(on_message)
        # ==================== 初始化线程 结束 ====================

        logging.info("✅ BacteriaAutoseg plugin loaded successfully.")

    # ==================== 重写 predict_shapes 方法 开始 ====================
    def predict_shapes(self, image, filename=None):
        logging.info("Triggering full auto-segmentation in a background thread.")

        # 如果上一个线程还在运行，则不启动新的
        if self.thread and self.thread.isRunning():
            logging.warning("A segmentation task is already running.")
            return

        # 2. 创建一个新线程和一个新工人
        self.thread = QThread()
        self.worker = SegmentationWorker(self.model, image, filename)
        self.worker.moveToThread(self.thread)

        # 3. 连接信号和槽
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_segmentation_finished)
        self.worker.error.connect(self.on_segmentation_error)

        # 4. 启动线程 (这个调用会立刻返回，不会阻塞UI)
        self.thread.start()
    # ==================== 重写 predict_shapes 方法 结束 ====================

    # ==================== 新增槽函数 开始 ====================
    def on_segmentation_finished(self, shapes):
        """当工作线程完成时，这个函数会在主线程中被调用"""
        logging.info("Main thread received segmentation results.")
        # 将结果包装成 AutoLabelingResult 并通过信号发送给UI
        self._prediction_done.emit(AutoLabelingResult(shapes, replace=True))
        self.cleanup_thread()

    def on_segmentation_error(self, error_message):
        """当工作线程出错时，这个函数会在主线程中被调用"""
        logging.error(f"Main thread received error: {error_message}")
        self._prediction_done.emit(AutoLabelingResult([], replace=False)) # 发送一个空结果
        self.cleanup_thread()
    
    def cleanup_thread(self):
        """清理线程资源"""
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None
    # ==================== 新增槽函数 结束 ====================

    def set_auto_labeling_reset_tracker(self):
        pass

    # Post-process 已经移到 Worker 中，可以保留或删除
    def post_process(self, mask: np.ndarray) -> list[Shape]:
        pass

    def split_mask(self, mask: np.ndarray, line_points: list) -> Union[list[np.ndarray], None]:
        return self.model.split_mask_with_line(mask, line_points)
        
    def unload(self):
        # 确保在卸载时停止任何正在运行的线程
        self.cleanup_thread()
        self.model = None
        logging.info("BacteriaAutoseg plugin unloaded.")
