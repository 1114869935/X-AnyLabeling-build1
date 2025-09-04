# 文件路径: anylabeling/services/auto_labeling/bacteria_autoseg.py
# 【修改后版本】

import logging
import traceback
import numpy as np
import cv2
from PyQt5 import QtCore

# ==================== 新增代码段 开始 ====================
import sys
import os

def get_resource_path(relative_path):
    """
    获取资源的绝对路径，无论是开发环境还是打包后的EXE都能用。
    """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 会创建一个临时文件夹，并把路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    else:
        # 在正常的开发环境中，我们假设脚本是从项目根目录运行的
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)
# ==================== 新增代码段 结束 ====================

from .model import Model
from .bacteria_onnx import BacteriaONNX
from .types import AutoLabelingResult
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img


class BacteriaAutoseg(Model):
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = []
        output_modes = {"polygon": "Polygon"}
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        super().__init__(config_path, on_message)
        
        # ==================== 修改代码段 开始 ====================
        # 1. 从 self.config (由基类加载的配置文件) 中获取相对路径
        encoder_rel_path = self.config.get("encoder_model_path")
        decoder_rel_path = self.config.get("decoder_model_path")
        input_size = self.config.get("input_size", 1024)  # 同时获取 input_size

        if not encoder_rel_path:
            raise ValueError("Config error: 'encoder_model_path' not found.")
        if not decoder_rel_path:
            raise ValueError("Config error: 'decoder_model_path' not found.")

        # 2. 使用我们新的辅助函数将相对路径转换为绝对路径
        encoder_model_abs_path = get_resource_path(encoder_rel_path)
        decoder_model_abs_path = get_resource_path(decoder_rel_path)

        # 3. 将两个【绝对路径】都传递给 BacteriaONNX 的构造函数
        #    我们稍后会修改 BacteriaONNX 来接收这两个路径
        self.model = BacteriaONNX(
            encoder_path=encoder_model_abs_path,
            decoder_path=decoder_model_abs_path,
            input_size=input_size
        )
        # ==================== 修改代码段 结束 ====================

        logging.info("✅ BacteriaAutoseg plugin loaded successfully.")

    # --- 以下代码保持不变 ---
    def set_auto_labeling_reset_tracker(self):
        pass

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        logging.info("Triggering full auto-segmentation for bacteria.")
        try:
            cv_image = qt_img_to_rgb_cv_img(image, filename)
            all_masks = self.model.predict_masks(cv_image)

            if all_masks.size == 0:
                logging.info("No masks were found by the model.")
                return AutoLabelingResult([], replace=True)

            all_shapes = []
            num_masks = all_masks.shape[0]
            logging.info(f"Generated {num_masks} masks. Post-processing...")

            for i in range(num_masks):
                shapes_from_mask = self.post_process(all_masks[i])
                all_shapes.extend(shapes_from_mask)
            
            logging.info(f"Finished. Found {len(all_shapes)} shapes.")
            return AutoLabelingResult(all_shapes, replace=True)

        except Exception as e:
            logging.error(f"Could not inference BacteriaAutoseg model: {e}")
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

    def post_process(self, mask: np.ndarray) -> list[Shape]:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        kernel = np.ones((7, 7), np.uint8)
        smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        shapes = []
        if self.output_mode == "polygon":
            for contour in contours:
                if cv2.contourArea(contour) < 10:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.004 * perimeter 
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                points = approx_contour.reshape(-1, 2).tolist()
                if len(points) < 3: continue
                
                shape = Shape(label="bacteria", shape_type="polygon")
                for point in points:
                    shape.add_point(QtCore.QPointF(int(point[0]), int(point[1])))
                shapes.append(shape)
        
        return shapes

    def split_mask(self, mask: np.ndarray, line_points: list) -> list[np.ndarray] | None:
        return self.model.split_mask_with_line(mask, line_points)
        
    def unload(self):
        self.model = None
        logging.info("BacteriaAutoseg plugin unloaded.")
