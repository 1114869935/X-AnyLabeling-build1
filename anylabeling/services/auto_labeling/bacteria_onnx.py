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
    Handles ONNX model loading and inference for bacteria segmentation using a robust
    affine transformation method for preprocessing and mask post-processing to ensure
    smooth and accurate mask boundaries.
    """

    def __init__(self, encoder_path: str, decoder_path: str, input_size: int = 1024):
        """
        Initializes the BacteriaONNX model loader.

        Args:
            encoder_path (str): The absolute path to the encoder ONNX model.
            decoder_path (str): The absolute path to the decoder ONNX model.
            input_size (int): The target square input size for the model.
        """
        self.input_size = input_size
        # --- Hyperparameters for segmentation quality ---
        self.points_per_side = 64
        self.pred_iou_thresh = 0.85
        self.stability_score_thresh = 0.90 # Optional, can be added if model provides stability scores
        self.min_mask_region_area = 120
        self.box_nms_thresh = 0.3
        self.max_area_ratio = 0.04
        self.min_circularity = 0.25
        self.mask_threshold = 0.5

        try:
            logging.info(f"Attempting to load ONNX models...")
            if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
                raise FileNotFoundError(f"Encoder or decoder model not found.")

            so = onnxruntime.SessionOptions()
            so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CPUExecutionProvider']

            self.enc_session = onnxruntime.InferenceSession(encoder_path, sess_options=so, providers=providers)
            self.dec_session = onnxruntime.InferenceSession(decoder_path, sess_options=so, providers=providers)

            self.enc_input_name = self.enc_session.get_inputs()[0].name
            logging.info(f"ONNX models loaded successfully using providers: {self.enc_session.get_providers()}")

            self._mask_input_template = np.zeros((1, 1, 256, 256), dtype=np.float32)
            self._has_mask_input_template = np.array([0.0], dtype=np.float32)

        except Exception as e:
            logging.error(f"Failed to load ONNX models. Error: {e}", exc_info=True)
            raise e

    def _preprocess_image(self, image_rgb: np.ndarray):
        """
        Preprocesses the image using an affine transformation to fit the model's input size.

        Returns:
            np.ndarray: The transformed image ready for the encoder.
            np.ndarray: The 2x3 affine transformation matrix used.
        """
        original_h, original_w, _ = image_rgb.shape
        scale = self.input_size / max(original_h, original_w)
        
        # The 2x3 affine transformation matrix for scaling
        M = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)
        
        # Apply the affine transformation
        transformed_image = cv2.warpAffine(
            image_rgb, M, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR
        )
        return transformed_image, M

    def _postprocess_mask(self, low_res_mask: np.ndarray, M: np.ndarray, original_size: tuple):
        """
        Transforms a single low-resolution mask back to the original image's coordinate space.
        
        Args:
            low_res_mask (np.ndarray): The 256x256 mask output from the model.
            M (np.ndarray): The forward affine transformation matrix used for preprocessing.
            original_size (tuple): The original image size (H, W).

        Returns:
            np.ndarray: The final probability mask on the original image.
        """
        # Upscale the low-resolution mask to the model's input size using cubic interpolation for smoothness
        prob_high_res = cv2.resize(
            low_res_mask, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC
        )
        
        # Invert the affine transformation matrix
        inv_M = cv2.invertAffineTransform(M)
        
        # Warp the high-resolution mask back to the original image dimensions
        prob_original_size = cv2.warpAffine(
            prob_high_res, inv_M, (original_size[1], original_size[0]), flags=cv2.INTER_LINEAR
        )
        
        return prob_original_size

    def predict_masks(self, cv_image: np.ndarray) -> np.ndarray:
        """
        Performs full inference on an image to get all bacteria masks.
        """
        try:
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            H, W, _ = image_rgb.shape

            # 1. Preprocess the image and get the transformation matrix
            transformed_image, M = self._preprocess_image(image_rgb)
            
            enc_input_3d = (transformed_image.astype(np.float32) / 255.0)
            enc_input = np.expand_dims(enc_input_3d, axis=0).transpose(0, 3, 1, 2).copy()
            image_embeddings = self.enc_session.run(None, {self.enc_input_name: enc_input})[0]

            # (Interest region calculation remains the same, as it operates on the original image)
            gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            roi_mask = self._find_petri_dish(gray_image)
            interest_mask = self._compute_interest_mask(gray_image)
            roi_interest = cv2.bitwise_and(roi_mask, interest_mask)
            grid_points = self._generate_grid_points(W, H, self.points_per_side)
            inside = roi_interest[grid_points[:, 1], grid_points[:, 0]] > 0
            points_inside_roi = grid_points[inside]

            if points_inside_roi.size == 0: return np.array([])
            
            # 2. Transform points to the model's input coordinate space
            points_hom = np.hstack([points_inside_roi, np.ones((len(points_inside_roi), 1))])
            transformed_points = (M @ points_hom.T).T

            # (Batching and inference logic remains similar)
            all_masks_data = []
            BATCH = 128
            for start in range(0, len(transformed_points), BATCH):
                batch_pts = transformed_points[start:start + BATCH]
                
                point_coords = np.expand_dims(batch_pts, axis=1) # Shape: (B, 1, 2)
                point_labels = np.full((len(batch_pts), 1), 1, dtype=np.float32) # Just foreground points

                feeds = {
                    "image_embeddings": image_embeddings,
                    "point_coords": point_coords.astype(np.float32),
                    "point_labels": point_labels.astype(np.float32),
                    "mask_input": np.repeat(self._mask_input_template, len(batch_pts), axis=0),
                    "has_mask_input": np.repeat(self._has_mask_input_template, len(batch_pts), axis=0),
                    "orig_im_size": np.array([H, W], dtype=np.float32),
                }
                masks, ious, low_res_logits = self.dec_session.run(None, feeds)
                
                for b in range(low_res_logits.shape[0]):
                    iou = float(ious[b, 0])
                    if iou < self.pred_iou_thresh: continue
                    
                    logits_256 = low_res_logits[b, 0]
                    prob_256 = _sigmoid(np.nan_to_num(np.clip(logits_256, -100, 100)))

                    # 3. Post-process the mask using the new affine method
                    prob_full = self._postprocess_mask(prob_256, M, (H, W))
                    
                    prob_full = cv2.GaussianBlur(prob_full, (3, 3), 0)
                    final_mask = (prob_full >= self.mask_threshold)

                    if np.count_nonzero(final_mask) <= self.min_mask_region_area: continue
                    all_masks_data.append({"mask": final_mask, "iou": iou})

            # (Final filtering and output stage remains the same)
            if not all_masks_data: return np.array([])
            base_filtered = self._base_postprocess(all_masks_data, self.min_mask_region_area, self.box_nms_thresh)
            if not base_filtered: return np.array([])
            advanced_filtered = self._advanced_postprocess(base_filtered, (H, W), self.max_area_ratio, self.min_circularity)

            final_masks_list = []
            roi_mask_bool = roi_mask.astype(bool, copy=False)
            for data in advanced_filtered:
                mask = np.logical_and(data['mask'], roi_mask_bool)
                if np.count_nonzero(mask) > self.min_mask_region_area:
                    final_masks_list.append(mask.astype(np.uint8))
            
            return np.stack(final_masks_list, axis=0) if final_masks_list else np.array([])

        except Exception as e:
            logging.error(f"Error during predict_masks: {e}", exc_info=True)
            return np.array([])

    # ===============================================
    #  HELPER FUNCTIONS (UNCHANGED FROM YOUR VERSION)
    # ===============================================
    def _find_petri_dish(self, gray_image: np.ndarray) -> np.ndarray:
        h, w = gray_image.shape
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 1.5)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h,
            param1=100, param2=80, minRadius=int(w * 0.2), maxRadius=int(w * 0.5)
        )
        roi_mask = np.zeros_like(gray_image, dtype=np.uint8)
        if circles is not None:
            cx, cy, r = np.uint16(np.around(circles))[0, 0]
            cv2.circle(roi_mask, (cx, cy), r, 255, -1)
        else:
            roi_mask.fill(255)
        return roi_mask

    def _compute_interest_mask(self, gray: np.ndarray, min_window: int = 9, var_thresh_percentile: float = 60.0) -> np.ndarray:
        H, W = gray.shape
        target = 512
        scale = float(target) / max(H, W)
        if scale < 1.0:
            ds_w, ds_h = int(round(W * scale)), int(round(H * scale))
            gray_small = cv2.resize(gray, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        else:
            gray_small = gray
        gray_small = cv2.GaussianBlur(gray_small, (3, 3), 0)
        lap = cv2.Laplacian(gray_small, ddepth=cv2.CV_32F, ksize=3)
        k = max(3, min_window | 1)
        mean = cv2.boxFilter(lap, ddepth=-1, ksize=(k, k), normalize=True)
        sq = lap * lap
        mean_of_sq = cv2.boxFilter(sq, ddepth=-1, ksize=(k, k), normalize=True)
        var_map = np.maximum(mean_of_sq - (mean * mean), 0.0)
        thr = np.percentile(var_map, var_thresh_percentile)
        interest_small = (var_map >= thr).astype(np.uint8) * 255
        interest_small = cv2.morphologyEx(interest_small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.resize(interest_small, (W, H), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else interest_small

    def _generate_grid_points(self, width: int, height: int, points_per_side: int) -> np.ndarray:
        grid_x = np.linspace(0, width - 1, points_per_side, dtype=np.int32)
        grid_y = np.linspace(0, height - 1, points_per_side, dtype=np.int32)
        xv, yv = np.meshgrid(grid_x, grid_y, indexing="xy")
        return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)

    def _filter_small_regions(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1: return mask
        keep_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        return np.isin(labels, keep_labels) if len(keep_labels) > 0 else np.zeros_like(mask, dtype=bool)

    def _calculate_box_iou(self, boxA, boxB) -> float:
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = float(boxA_area + boxB_area - inter_area)
        return inter_area / union_area if union_area > 0 else 0.0

    def _base_postprocess(self, masks_data: list, min_mask_region_area: int, box_nms_thresh: float) -> list:
        filtered = []
        for data in masks_data:
            cleaned_mask = self._filter_small_regions(data["mask"], min_mask_region_area)
            area = np.count_nonzero(cleaned_mask)
            if area == 0: continue
            rows, cols = np.any(cleaned_mask, axis=1), np.any(cleaned_mask, axis=0)
            if not (rows.any() and cols.any()): continue
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            filtered.append({"mask": cleaned_mask, "area": int(area), "bbox": [x_min, y_min, x_max, y_max], "iou": data["iou"]})
        if not filtered: return []
        filtered.sort(key=lambda x: x["iou"], reverse=True)
        final_masks = []
        while filtered:
            best = filtered.pop(0)
            final_masks.append(best)
            filtered = [o for o in filtered if self._calculate_box_iou(best["bbox"], o["bbox"]) < box_nms_thresh]
        return final_masks

    def _advanced_postprocess(self, masks_data: list, original_size: tuple, max_area_ratio: float, min_circularity: float) -> list:
        reference_area = float(original_size[0] * original_size[1])
        advanced_filtered = []
        for data in masks_data:
            if (data["area"] / reference_area) > max_area_ratio: continue
            contours, _ = cv2.findContours(data["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * data["area"] / (perimeter ** 2)
            if circularity < min_circularity: continue
            advanced_filtered.append(data)
        return advanced_filtered
