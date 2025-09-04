# 文件路径: anylabeling/services/auto_labeling/bacteria_onnx.py
# 【最终修正版 - 适用于PyInstaller打包 | CPU-only + 早期过滤 + 解码批处理 + 复杂度优化】

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
    """
    Handles ONNX model loading and inference for bacteria segmentation.
    Modified to accept direct, absolute paths for encoder and decoder models,
    making it compatible with PyInstaller-packaged applications.
    """

    # ==================== 修改后的 __init__ 方法 ====================
    def __init__(self, encoder_path: str, decoder_path: str, input_size: int = 1024):
        """
        Initializes the BacteriaONNX model loader.

        Args:
            encoder_path (str): The absolute path to the encoder ONNX model.
            decoder_path (str): The absolute path to the decoder ONNX model.
            input_size (int): The target input size for the model.
        """
        self.input_size = input_size
        self.points_per_side = 64
        self.pred_iou_thresh = 0.85
        self.min_mask_region_area = 120
        self.box_nms_thresh = 0.3
        self.max_area_ratio = 0.04
        self.min_circularity = 0.25
        self.mask_threshold = 0.5

        try:
            logging.info(f"Attempting to load encoder model from: {encoder_path}")
            logging.info(f"Attempting to load decoder model from: {decoder_path}")

            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder model not found at resolved path: {encoder_path}")
            if not os.path.exists(decoder_path):
                raise FileNotFoundError(f"Decoder model not found at resolved path: {decoder_path}")

            # ---- ONNX Runtime：CPU-only 性能优化设置 ----
            so = onnxruntime.SessionOptions()
            so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            try:
                cpu_cnt = os.cpu_count() or 4
                # 经验值：避免开过多线程导致上下文切换开销
                so.intra_op_num_threads = min(4, cpu_cnt)
                so.inter_op_num_threads = 1
                so.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            except Exception:
                pass

            providers = ['CPUExecutionProvider']  # 强制 CPU

            self.enc_session = onnxruntime.InferenceSession(encoder_path, sess_options=so, providers=providers)
            self.dec_session = onnxruntime.InferenceSession(decoder_path, sess_options=so, providers=providers)

            self.enc_input_name = self.enc_session.get_inputs()[0].name
            logging.info(f"ONNX models loaded successfully, using providers: {self.enc_session.get_providers()}")

            # 记录 decoder 输入名（避免硬编码），并预创建可复用模板
            self.dec_input_names = [i.name for i in self.dec_session.get_inputs()]
            self._mask_input_template = np.zeros((1, 1, 256, 256), dtype=np.float32)
            self._has_mask_input_template = np.array([0.0], dtype=np.float32)

        except Exception as e:
            logging.error(f"Failed to load ONNX models. Error: {e}", exc_info=True)
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
            enc_input = np.expand_dims(enc_input_3d, axis=0).transpose(0, 3, 1, 2).copy()

            # Encoder 前向
            image_embeddings = self.enc_session.run(None, {self.enc_input_name: enc_input})[0]

            # ---- ROI：培养皿检测（一次）+ 早期兴趣区域过滤 ----
            gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            roi_mask = self._find_petri_dish(gray_image)
            interest_mask = self._compute_interest_mask(gray_image, min_window=9, var_thresh_percentile=60.0)
            roi_interest = cv2.bitwise_and(roi_mask, interest_mask)

            # ---- 网格点向量化筛选 ----
            grid_points = self._generate_grid_points(W, H, self.points_per_side)
            pts_x = grid_points[:, 0]
            pts_y = grid_points[:, 1]
            inside = roi_interest[pts_y, pts_x] > 0
            points_inside_roi = grid_points[inside]

            if points_inside_roi.size == 0:
                return np.array([])

            # 可选：上限控制，避免极端情况下点过多压垮 CPU
            MAX_POINTS = 3000
            if len(points_inside_roi) > MAX_POINTS:
                idx = np.linspace(0, len(points_inside_roi) - 1, MAX_POINTS, dtype=np.int32)
                points_inside_roi = points_inside_roi[idx]

            all_masks_data = []

            # ---- 解码批处理：减少 session.run 次数（如模型不支持，则回退单点） ----
            BATCH = 128
            try:
                for start in range(0, len(points_inside_roi), BATCH):
                    batch_pts = points_inside_roi[start:start + BATCH].astype(np.float32)
                    tx = batch_pts[:, 0] * scale
                    ty = batch_pts[:, 1] * scale

                    point_coords = np.stack([
                        np.stack([tx, ty], axis=1),
                        np.zeros_like(batch_pts, dtype=np.float32)
                    ], axis=1)  # (B,2,2)
                    point_labels = np.stack([np.array([1.0, -1.0], dtype=np.float32)] * len(batch_pts), axis=0)  # (B,2)
                    mask_input = np.repeat(self._mask_input_template, repeats=len(batch_pts), axis=0)
                    has_mask_input = np.repeat(self._has_mask_input_template, repeats=len(batch_pts), axis=0)
                    orig_im_size = np.repeat(np.array([[H, W]], dtype=np.float32), repeats=len(batch_pts), axis=0)

                    feeds = {
                        "image_embeddings": image_embeddings,
                        "point_coords": point_coords.astype(np.float32, copy=False),
                        "point_labels": point_labels,
                        "mask_input": mask_input,
                        "has_mask_input": has_mask_input,
                        "orig_im_size": orig_im_size,
                    }

                    masks, ious, low_res_logits = self.dec_session.run(None, feeds)

                    # 合并插值：256 -> (new_w,new_h) -> (W,H)；轻量 3x3 平滑
                    C = low_res_logits.shape[1]
                    for b in range(low_res_logits.shape[0]):
                        for i in range(C):
                            iou = float(ious[b, i])
                            if iou < self.pred_iou_thresh:
                                continue
                            logits_256 = low_res_logits[b, i]
                            prob_256 = _sigmoid(np.nan_to_num(np.clip(logits_256, -100, 100)))
                            prob_nopad = cv2.resize(prob_256, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                            prob_full = cv2.resize(prob_nopad, (W, H), interpolation=cv2.INTER_LINEAR)
                            prob_full = cv2.GaussianBlur(prob_full, (3, 3), 0)
                            final_mask = (prob_full >= self.mask_threshold)

                            # —— 立即面积过滤（你提出的改动）：太小的不加入 ——
                            if np.count_nonzero(final_mask) <= self.min_mask_region_area:
                                continue

                            all_masks_data.append({"mask": final_mask, "iou": iou})
            except Exception:
                # 批处理不支持时回退逐点
                for point in points_inside_roi:
                    results_for_point = self._process_single_point(point, image_embeddings, scale, H, W, new_h, new_w)
                    if results_for_point:
                        all_masks_data.extend(results_for_point)

            if not all_masks_data:
                return np.array([])

            # 基础后处理（小区域过滤 + NMS）
            base_filtered = self._base_postprocess(all_masks_data, self.min_mask_region_area, self.box_nms_thresh)
            if not base_filtered:
                return np.array([])

            # 几何约束过滤（面积占比 + 圆度）
            advanced_filtered = self._advanced_postprocess(base_filtered, (H, W), self.max_area_ratio, self.min_circularity)

            # ROI 裁剪
            final_masks_list = []
            roi_mask_bool = roi_mask.astype(bool, copy=False)
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
        """
        Processes a single point prompt to generate potential masks.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*overflow.*')
            warnings.filterwarnings('ignore', message='.*invalid value.*')

            px, py = point_data
            tx, ty = px * scale, py * scale

            point_coords = np.array([[[tx, ty], [0.0, 0.0]]], dtype=np.float32)
            point_labels = np.array([[1.0, -1.0]], dtype=np.float32)
            mask_input = self._mask_input_template.copy()
            has_mask_input = self._has_mask_input_template.copy()
            orig_im_size = np.array([H, W], dtype=np.float32)

            feeds = {
                "image_embeddings": image_embeddings,
                "point_coords": point_coords,
                "point_labels": point_labels,
                "mask_input": mask_input,
                "has_mask_input": has_mask_input,
                "orig_im_size": orig_im_size
            }

            masks, ious, low_res_logits = self.dec_session.run(None, feeds)

            results_for_point = []
            for i in range(masks.shape[1]):
                iou = float(ious[0, i])
                if iou < self.pred_iou_thresh:
                    continue

                logits_256 = low_res_logits[0, i]
                prob_256 = _sigmoid(np.nan_to_num(np.clip(logits_256, -100, 100)))

                # 合并插值路径
                prob_nopad = cv2.resize(prob_256, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                prob_full = cv2.resize(prob_nopad, (W, H), interpolation=cv2.INTER_LINEAR)
                prob_full = cv2.GaussianBlur(prob_full, (3, 3), 0)

                final_mask = (prob_full >= self.mask_threshold)

                # —— 立即面积过滤：太小直接丢弃 ——
                if np.count_nonzero(final_mask) <= self.min_mask_region_area:
                    continue

                results_for_point.append({"mask": final_mask, "iou": iou})

            return results_for_point

    def _find_petri_dish(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Finds the petri dish in the image to create a region of interest (ROI).
        """
        h, w = gray_image.shape
        # 更轻的预模糊，降低 Hough 前成本
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 1.5)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=h,
            param1=100, param2=80,
            minRadius=int(w * 0.2), maxRadius=int(w * 0.5)
        )

        roi_mask = np.zeros_like(gray_image, dtype=np.uint8)
        if circles is not None:
            cx, cy, r = np.uint16(np.around(circles))[0, 0]
            cv2.circle(roi_mask, (cx, cy), r, 255, -1)
        else:
            roi_mask.fill(255)
        return roi_mask

    def _compute_interest_mask(self, gray: np.ndarray, min_window: int = 9, var_thresh_percentile: float = 60.0) -> np.ndarray:
        """
        计算兴趣区域掩码：用 Laplacian 方差近似局部纹理强度，阈值采用分位数，返回与 gray 同尺寸的 uint8 {0,255} 掩码。
        """
        H, W = gray.shape
        target = 512
        scale = float(target) / max(H, W)
        if scale < 1.0:
            ds_w, ds_h = int(round(W * scale)), int(round(H * scale))
            gray_small = cv2.resize(gray, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        else:
            gray_small = gray

        gray_small = cv2.GaussianBlur(gray_small, (3, 3), 0)

        # Laplacian 响应
        lap = cv2.Laplacian(gray_small, ddepth=cv2.CV_32F, ksize=3)

        # 局部均值与方差的快速近似（盒滤）
        k = max(3, min_window | 1)
        mean = cv2.boxFilter(lap, ddepth=-1, ksize=(k, k), normalize=True)
        mean_sq = mean * mean
        sq = lap * lap
        mean_of_sq = cv2.boxFilter(sq, ddepth=-1, ksize=(k, k), normalize=True)
        var_map = np.maximum(mean_of_sq - mean_sq, 0.0)

        thr = np.percentile(var_map, var_thresh_percentile)
        interest_small = (var_map >= thr).astype(np.uint8) * 255
        interest_small = cv2.morphologyEx(interest_small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        if (gray_small.shape[0], gray_small.shape[1]) != (H, W):
            interest = cv2.resize(interest_small, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            interest = interest_small
        return interest

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
            area = np.count_nonzero(cleaned_mask)
            if area == 0:
                continue

            rows = np.any(cleaned_mask, axis=1)
            cols = np.any(cleaned_mask, axis=0)
            if not (rows.any() and cols.any()):
                continue

            y_idx = np.flatnonzero(rows)
            x_idx = np.flatnonzero(cols)
            y_min, y_max = y_idx[0], y_idx[-1]
            x_min, x_max = x_idx[0], x_idx[-1]

            filtered.append({
                "mask": cleaned_mask,
                "area": int(area),
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "iou": float(data["iou"])
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

            circularity = 4 * np.pi * data["area"] / (perimeter ** 2)
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
