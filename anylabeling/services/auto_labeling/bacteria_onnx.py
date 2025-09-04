# 文件路径: anylabeling/services/auto_labeling/bacteria_onnx.py
# 【最终修正版 - 适用于PyInstaller打包】

import cv2
import numpy as np
import onnxruntime
import logging
import os
import warnings

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Helper function to compute sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

class BacteriaONNX:
    # 【使用这个"CPU强制+详细日志"的 __init__ 方法来替换您原来的版本】
    def __init__(self, encoder_path: str, decoder_path: str, input_size: int = 1024):
        print("--- [DEBUG] Initializing BacteriaONNX (CPU Force Mode) ---")
        print(f"[DEBUG] Received encoder_path: {encoder_path}")
        print(f"[DEBUG] Received decoder_path: {decoder_path}")

        self.input_size = input_size
        self.points_per_side = 64
        self.pred_iou_thresh = 0.85
        self.min_mask_region_area = 120
        self.box_nms_thresh = 0.3
        self.max_area_ratio = 0.04
        self.min_circularity = 0.25
        self.mask_threshold = 0.5

        try:
            print(f"[DEBUG] Checking if encoder file exists at the path...")
            if not os.path.exists(encoder_path):
                print(f"[FATAL ERROR] Encoder model not found at resolved path: {encoder_path}")
                raise FileNotFoundError(f"Encoder model not found at: {encoder_path}")
            print("[DEBUG] Encoder file found!")

            print(f"[DEBUG] Checking if decoder file exists at the path...")
            if not os.path.exists(decoder_path):
                print(f"[FATAL ERROR] Decoder model not found at resolved path: {decoder_path}")
                raise FileNotFoundError(f"Decoder model not found at: {decoder_path}")
            print("[DEBUG] Decoder file found!")

            so = onnxruntime.SessionOptions()
            
            # ======================= 【核心修改点】 =======================
            # 我们删除了 'CUDAExecutionProvider'，强制 ONNX 只使用 CPU。
            # 这可以绕过所有 GPU 驱动问题，是解决“卡死”问题的最有效手段。
            providers = ['CPUExecutionProvider']
            print(f"[DEBUG] Forcing ONNX Runtime to use providers: {providers}")
            # =============================================================
            
            print(f"[DEBUG] Attempting to load ENCODER model into ONNX Runtime Session...")
            self.enc_session = onnxruntime.InferenceSession(encoder_path, sess_options=so, providers=providers)
            print("[DEBUG] ENCODER model loaded successfully!")

            print(f"[DEBUG] Attempting to load DECODER model into ONNX Runtime Session...")
            self.dec_session = onnxruntime.InferenceSession(decoder_path, sess_options=so, providers=providers)
            print("[DEBUG] DECODER model loaded successfully!")

            self.enc_input_name = self.enc_session.get_inputs()[0].name
            print(f"[DEBUG] ONNX models initialized, using providers: {self.enc_session.get_providers()}")
            print("--- [DEBUG] BacteriaONNX Initialization Complete ---")

        except Exception as e:
            import traceback
            print("[FATAL ERROR] An exception occurred during ONNX model loading!")
            print(str(e))
            traceback.print_exc()
            input("Press Enter to exit...") 
            raise e
    # ==================== __init__ 方法修改结束 ====================

    def predict_masks(self, cv_image: np.ndarray) -> np.ndarray:
        """
        Performs full inference on an image to get all bacteria masks.
        """
        try:
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            H, W, _ = image_rgb.shape
            
            padded_img, _, (new_h, new_w), scale = self._resize_longest_side_and_pad(image_rgb, self.input_size)
            
            enc_input_3d = (padded_img.astype(np.float32) / 255.0)
            enc_input = np.expand_dims(enc_input_3d, axis=0).transpose(0, 3, 1, 2)
            
            image_embeddings = self.enc_session.run(None, {self.enc_input_name: enc_input})[0]
            
            gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            roi_mask = self._find_petri_dish(gray_image)
            
            grid_points = self._generate_grid_points(W, H, self.points_per_side)
            points_inside_roi = grid_points[roi_mask[grid_points[:, 1], grid_points[:, 0]] > 0]
            
            if len(points_inside_roi) == 0: 
                return np.array([])
            
            all_masks_data = []
            for point in points_inside_roi:
                results_for_point = self._process_single_point(point, image_embeddings, scale, H, W, new_h, new_w)
                if results_for_point:
                    all_masks_data.extend(results_for_point)
            
            if not all_masks_data:
                return np.array([])
            
            base_filtered = self._base_postprocess(all_masks_data, self.min_mask_region_area, self.box_nms_thresh)
            
            if not base_filtered:
                return np.array([])
            
            advanced_filtered = self._advanced_postprocess(base_filtered, (H, W), self.max_area_ratio, self.min_circularity)
            
            final_masks_list = []
            roi_mask_bool = roi_mask.astype(bool)
            for data in advanced_filtered:
                mask = np.logical_and(data['mask'], roi_mask_bool)
                if np.count_nonzero(mask) > self.min_mask_region_area:
                    final_masks_list.append(mask.astype(np.uint8))
            
            if not final_masks_list:
                return np.array([])
                
            return np.stack(final_masks_list, axis=0)

        except Exception as e:
            logging.error(f"Error during predict_masks: {e}", exc_info=True)
            return np.array([])

    def _process_single_point(self, point_data, image_embeddings, scale, H, W, new_h, new_w):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*overflow.*')
            warnings.filterwarnings('ignore', message='.*invalid value.*')
            px, py = point_data
            tx, ty = px * scale, py * scale
            point_coords = np.array([[[tx, ty], [0.0, 0.0]]], dtype=np.float32)
            point_labels = np.array([[1.0, -1.0]], dtype=np.float32)
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0.0], dtype=np.float32)
            orig_im_size = np.array([H, W], dtype=np.float32)
            feeds = {"image_embeddings": image_embeddings, "point_coords": point_coords, "point_labels": point_labels, "mask_input": mask_input, "has_mask_input": has_mask_input, "orig_im_size": orig_im_size}
            masks, ious, low_res_logits = self.dec_session.run(None, feeds)
            results_for_point = []
            for i in range(masks.shape[1]):
                iou = float(ious[0, i])
                if iou < self.pred_iou_thresh: continue
                logits_256 = low_res_logits[0, i]
                prob_256 = _sigmoid(np.nan_to_num(np.clip(logits_256, -100, 100)))
                prob_1024 = cv2.resize(prob_256, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
                prob_padded = prob_1024[:new_h, :new_w]
                prob_blurred = cv2.GaussianBlur(prob_padded, (5, 5), 0)
                upscale_factor = 2
                prob_upscaled = cv2.resize(prob_blurred, (W * upscale_factor, H * upscale_factor), interpolation=cv2.INTER_CUBIC)
                mask_upscaled = (prob_upscaled >= self.mask_threshold)
                final_mask = cv2.resize(mask_upscaled.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

                # ======================= 【核心修复点】 =======================
                # 在这里立即过滤掉过小的、无用的 mask，而不是等收集完所有结果之后。
                # 这将极大地减少内存占用和后续处理的计算量。
                if np.count_nonzero(final_mask) > self.min_mask_region_area:
                    results_for_point.append({"mask": final_mask, "iou": iou})
                # =============================================================
            
            return results_for_point

    def _find_petri_dish(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Finds the petri dish in the image to create a region of interest (ROI).
        """
        h, w = gray_image.shape
        blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h, param1=100, param2=80, minRadius=int(w * 0.2), maxRadius=int(w * 0.5))
        
        roi_mask = np.zeros_like(gray_image, dtype=np.uint8)
        if circles is not None:
            cx, cy, r = np.uint16(np.around(circles))[0, 0]
            cv2.circle(roi_mask, (cx, cy), r, 255, -1)
        else:
            # If no circle is found, assume the whole image is the ROI.
            roi_mask.fill(255)
            
        return roi_mask

    def _generate_grid_points(self, width: int, height: int, points_per_side: int) -> np.ndarray:
        """
        Generates a grid of points over the image.
        """
        grid_x = np.linspace(0, width - 1, points_per_side, dtype=np.float32)
        grid_y = np.linspace(0, height - 1, points_per_side, dtype=np.float32)
        xv, yv = np.meshgrid(grid_x, grid_y, indexing="xy")
        return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.int32)
    
    def _filter_small_regions(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Removes small, disconnected regions from a binary mask.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return mask
        
        keep_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        return np.isin(labels, keep_labels) if len(keep_labels) > 0 else np.zeros_like(mask, dtype=bool)

    def _calculate_box_iou(self, boxA, boxB) -> float:
        """
        Calculates Intersection over Union (IoU) for two bounding boxes.
        """
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        
        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        union_area = float(boxA_area + boxB_area - inter_area)
        return inter_area / union_area if union_area > 0 else 0.0

    def _base_postprocess(self, masks_data: list, min_mask_region_area: int, box_nms_thresh: float) -> list:
        """
        Performs basic post-processing: filters small masks and applies Non-Maximum Suppression.
        """
        filtered = []
        for data in masks_data:
            cleaned_mask = self._filter_small_regions(data["mask"], min_mask_region_area)
            if np.count_nonzero(cleaned_mask) == 0:
                continue
            
            rows, cols = np.any(cleaned_mask, axis=1), np.any(cleaned_mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            filtered.append({
                "mask": cleaned_mask,
                "area": np.count_nonzero(cleaned_mask),
                "bbox": [x_min, y_min, x_max, y_max],
                "iou": data["iou"]
            })
            
        if not filtered:
            return []
            
        filtered.sort(key=lambda x: x["iou"], reverse=True)
        
        final_masks = []
        while filtered:
            best = filtered.pop(0)
            final_masks.append(best)
            filtered = [o for o in filtered if self._calculate_box_iou(best["bbox"], o["bbox"]) < box_nms_thresh]
            
        return final_masks

    def _advanced_postprocess(self, masks_data: list, original_size: tuple, max_area_ratio: float, min_circularity: float) -> list:
        """
        Performs advanced filtering based on mask properties like area ratio and circularity.
        """
        reference_area = float(original_size[0] * original_size[1])
        advanced_filtered = []
        
        for data in masks_data:
            if (data["area"] / reference_area) > max_area_ratio:
                continue
            
            contours, _ = cv2.findContours(data["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * data["area"] / (perimeter**2)
            if circularity < min_circularity:
                continue
                
            advanced_filtered.append(data)
            
        return advanced_filtered

    def _resize_longest_side_and_pad(self, img: np.ndarray, target_length: int):
        """
        Resizes the image's longest side to a target length and pads to a square.
        """
        H, W = img.shape[:2]
        scale = float(target_length) / max(H, W)
        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        padded = np.zeros((target_length, target_length, 3), dtype=resized.dtype)
        padded[:new_h, :new_w] = resized
        
        return padded, (H, W), (new_h, new_w), scale
