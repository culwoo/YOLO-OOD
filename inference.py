"""
YOLO + OOD Detection Inference Module (CPU version)
"""

import torch
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.instance import Instances
import cv2

from utils import apply_clahe_cpu, draw_bboxes, draw_ood_warning


class YOLOOODDetector:
    """
    Combined YOLO object detection and Mahalanobis-based OOD detection
    CPU-only version for local inference
    """

    def __init__(self, model_path="models/best.pt", stats_path="models/id_stats.pkl"):
        """
        Initialize detector with model and OOD statistics

        Args:
            model_path: Path to YOLO model weights
            stats_path: Path to OOD statistics pickle file
        """
        # Prefer GPU when available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load YOLO model
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load OOD statistics
        print(f"Loading OOD statistics from {stats_path}...")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        self.mu = torch.from_numpy(stats["mu"]).float().to(self.device)
        self.cov_inv = torch.from_numpy(stats["cov_inv"]).float().to(self.device)
        self.threshold = stats["threshold"]

        print(f"OOD threshold: {self.threshold:.4f}")

        # LetterBox transform (matches training)
        self.letterbox = LetterBox(new_shape=(544, 960), auto=False)
        self.model_input_shape = (544, 960)

        # Neck layer for feature extraction
        self.neck_layer = self.model.model.model[22]
        self.features = None

        print("Model initialized successfully!")

    def _hook_fn(self, module, input, output):
        """Hook function to extract features from neck layer"""
        # Global Average Pooling
        self.features = torch.mean(output, dim=(2, 3))

    def _mahalanobis_distance(self, x):
        """
        Calculate Mahalanobis distance

        Args:
            x: Feature vector (C,)

        Returns:
            distance: Mahalanobis distance (scalar)
        """
        diff = x - self.mu
        left = torch.matmul(diff, self.cov_inv)
        distance = torch.sqrt(torch.sum(left * diff))
        return distance.item()

    def preprocess_image(self, image, use_clahe=False):
        """
        Preprocess image for inference

        Args:
            image: PIL Image or numpy array (RGB)
            use_clahe: Whether to apply CLAHE preprocessing

        Returns:
            processed_img: Preprocessed image tensor (1, 3, H, W)
            original_img: Original image in BGR format for visualization
            original_shape: (H, W) original image size
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()

        # Ensure RGB format
        if img_array.shape[-1] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        original_shape = img_array.shape[:2]

        # Save original for visualization (convert to BGR for OpenCV)
        original_img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Apply CLAHE if requested
        if use_clahe:
            img_array = apply_clahe_cpu(img_array)

        # Apply LetterBox (same as training)
        # Create empty Instances object for inference (no ground truth boxes)
        empty_instances = Instances(
            bboxes=np.zeros((0, 4), dtype=np.float32),
            segments=np.zeros((0, 0, 2), dtype=np.float32),
            bbox_format="xywh"
        )

        # Create sample dict for LetterBox
        sample = {
            "img": img_array,
            "cls": np.array([]),
            "instances": empty_instances,
            "ori_shape": img_array.shape[:2],
            "resized_shape": img_array.shape[:2],
            "batch_idx": 0
        }

        sample = self.letterbox(sample)

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(sample["img"]).permute(2, 0, 1).contiguous()
        img_tensor = img_tensor.unsqueeze(0).float() / 255.0  # Add batch dim and normalize

        return img_tensor, original_img_bgr, original_shape

    def detect(self, image, use_clahe=False, conf_threshold=0.001, ood_threshold=None):
        """
        Perform YOLO detection + OOD detection on a single image

        Args:
            image: PIL Image or numpy array (RGB)
            use_clahe: Whether to apply CLAHE preprocessing
            conf_threshold: Confidence threshold for YOLO detection
            ood_threshold: Custom OOD threshold (uses default if None)

        Returns:
            result_image: Visualized result image (BGR format for OpenCV/Gradio)
            ood_score: Mahalanobis distance score
            is_ood: Boolean indicating if image is OOD
            detection_results: YOLO detection results
        """
        # Use custom threshold or default
        threshold_to_use = ood_threshold if ood_threshold is not None else self.threshold

        # Preprocess
        img_tensor, original_img_bgr, original_shape = self.preprocess_image(image, use_clahe)
        img_tensor = img_tensor.to(self.device)

        # Register hook for feature extraction
        handle = self.neck_layer.register_forward_hook(self._hook_fn)

        with torch.no_grad():
            # YOLO detection
            results = self.model(img_tensor, conf=conf_threshold, verbose=False)

            # Extract features (from hook)
            if self.features is not None:
                feats = self.features
            else:
                # Fallback: manually extract features
                feat_map = self.model.model.model[:23](img_tensor)
                feats = torch.mean(feat_map, dim=(2, 3))

            # Calculate Mahalanobis distance
            feats_squeeze = feats.squeeze(0)  # Remove batch dimension
            ood_score = self._mahalanobis_distance(feats_squeeze)

        # Remove hook
        handle.remove()

        # Determine if OOD (using custom or default threshold)
        is_ood = ood_score > threshold_to_use

        # Visualize results
        result_image = self._visualize_results(
            original_img_bgr,
            results,
            original_shape,
            is_ood,
            ood_score,
            conf_threshold,
            threshold_to_use
        )

        return result_image, ood_score, is_ood, results

    def _visualize_results(self, img, results, original_shape, is_ood, ood_score, conf_threshold, ood_threshold):
        """
        Visualize detection results

        Args:
            img: Original image (BGR)
            results: YOLO results
            original_shape: (H, W) original image size
            is_ood: Whether image is OOD
            ood_score: Mahalanobis distance score
            conf_threshold: Confidence threshold
            ood_threshold: OOD threshold used for detection

        Returns:
            result_img: Visualized result image (BGR)
        """
        # Draw bounding boxes
        img_with_boxes = draw_bboxes(
            img,
            results,
            original_shape=original_shape,
            model_input_shape=self.model_input_shape,
            conf_threshold=conf_threshold
        )

        # Draw OOD warning if detected
        result_img = draw_ood_warning(img_with_boxes, is_ood, alpha=0.5)

        # Add score information (bottom-left corner) - DISABLED
        # H, W = img.shape[:2]
        # font_scale = max(0.6, min(H, W) / 1500)
        # thickness = max(2, int(font_scale * 3))

        # info_text = f"Score: {ood_score:.2f} | Threshold: {ood_threshold:.2f}"
        # status_text = "OOD" if is_ood else "ID"
        # status_color = (0, 0, 255) if is_ood else (0, 255, 0)

        # y_offset = H - int(H * 0.05)

        # # Info text background
        # (text_w, text_h), _ = cv2.getTextSize(
        #     info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        # )
        # cv2.rectangle(result_img,
        #              (15, y_offset - 55),
        #              (25 + text_w, y_offset + 10),
        #              (0, 0, 0), -1)

        # cv2.putText(result_img, info_text, (20, y_offset - 40),
        #            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        # cv2.putText(result_img, f"Status: {status_text}", (20, y_offset),
        #            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, status_color, thickness)

        return result_img


def create_detector():
    """
    Factory function to create detector instance
    Useful for lazy loading in Gradio
    """
    return YOLOOODDetector()
