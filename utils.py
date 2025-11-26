"""
Utility functions for YOLO + OOD detection visualization
"""

import cv2
import numpy as np


def apply_clahe_cpu(img_rgb):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing
    CPU version only

    Args:
        img_rgb: (H, W, 3) RGB image array

    Returns:
        clahe_rgb: (H, W, 3) RGB image with CLAHE applied
    """
    clahe_cpu = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Convert to YCrCb color space
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

    # Apply CLAHE to Y channel only
    img_ycrcb[:, :, 0] = clahe_cpu.apply(img_ycrcb[:, :, 0])

    # Convert back to RGB
    clahe_rgb = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)

    return clahe_rgb


def create_ood_warning_overlay(img_shape, intensity=0.3, border_width=None):
    """
    Create navigation-style OOD warning overlay (auto-scaled to image size)

    Args:
        img_shape: (H, W, C) or (H, W)
        intensity: Red color intensity (0~1)
        border_width: Border thickness in pixels (auto if None)

    Returns:
        overlay: (H, W, 3) uint8 array
    """
    H, W = img_shape[:2]

    # Auto-scale border width based on image size (2% of min dimension)
    if border_width is None:
        border_width = max(20, int(min(H, W) * 0.02))

    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    # Red borders (top, bottom, left, right)
    red_color = int(255 * intensity)
    overlay[:border_width, :] = [0, 0, red_color]  # Top
    overlay[-border_width:, :] = [0, 0, red_color]  # Bottom
    overlay[:, :border_width] = [0, 0, red_color]  # Left
    overlay[:, -border_width:] = [0, 0, red_color]  # Right

    # Emphasized corners - REMOVED
    # corner_size = border_width * 2
    # overlay[:corner_size, :corner_size] = [0, 0, 255]  # Top-left
    # overlay[:corner_size, -corner_size:] = [0, 0, 255]  # Top-right
    # overlay[-corner_size:, :corner_size] = [0, 0, 255]  # Bottom-left
    # overlay[-corner_size:, -corner_size:] = [0, 0, 255]  # Bottom-right

    return overlay


def draw_ood_warning(img, ood_detected, alpha=0.5):
    """
    Draw OOD warning on image (auto-scaled to original size)

    Args:
        img: (H, W, 3) BGR image
        ood_detected: bool, whether OOD is detected
        alpha: Overlay transparency

    Returns:
        img_with_warning: (H, W, 3) BGR image
    """
    if not ood_detected:
        return img

    # Create overlay directly in BGR space so that red borders appear correctly
    overlay = create_ood_warning_overlay(img.shape, intensity=0.6)

    # Blend overlay with image
    img_with_warning = cv2.addWeighted(img, 1, overlay, alpha, 0)

    # Add "OOD DETECTED!" text
    H, W = img.shape[:2]
    font_scale = max(1.0, min(H, W) / 1000)  # Dynamic font size
    thickness = max(2, int(font_scale * 2))

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "WARNING: OOD DETECTED!"

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = (W - text_w) // 2
    y = int(H * 0.08)  # Top 8% position

    # Text background (black rectangle)
    padding = 15
    cv2.rectangle(img_with_warning,
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + padding),
                  (0, 0, 0), -1)

    # Text (red in BGR)
    cv2.putText(img_with_warning, text, (x, y),
                font, font_scale, (0, 0, 255), thickness)

    return img_with_warning


def draw_bboxes(img, results, original_shape, model_input_shape=(544, 960), conf_threshold=0.001):
    """
    Draw YOLO bounding boxes with coordinate scaling correction

    Args:
        img: (H, W, 3) BGR image (original size)
        results: YOLO results object
        original_shape: (H, W) original image size
        model_input_shape: (H, W) model input size (544, 960)
        conf_threshold: Confidence threshold

    Returns:
        img_with_boxes: (H, W, 3) BGR image with bounding boxes
    """
    img_with_boxes = img.copy()
    orig_h, orig_w = original_shape
    model_h, model_w = model_input_shape

    # Calculate scale for LetterBox (maintains aspect ratio)
    scale = min(model_h / orig_h, model_w / orig_w)

    # Actual resized size
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)

    # Calculate padding
    pad_h = (model_h - new_h) / 2
    pad_w = (model_w - new_w) / 2

    if results and len(results) > 0:
        result = results[0]  # First image result

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.cpu().numpy()

            for box in boxes:
                conf = box.conf[0]
                if conf < conf_threshold:
                    continue

                cls_id = int(box.cls[0])

                # Model output coordinates (960x544 basis)
                x1_model, y1_model, x2_model, y2_model = box.xyxy[0]

                # Remove padding
                x1_unpad = x1_model - pad_w
                y1_unpad = y1_model - pad_h
                x2_unpad = x2_model - pad_w
                y2_unpad = y2_model - pad_h

                # Scale back to original size
                x1_orig = int(x1_unpad / scale)
                y1_orig = int(y1_unpad / scale)
                x2_orig = int(x2_unpad / scale)
                y2_orig = int(y2_unpad / scale)

                # Clip to image boundaries
                x1_orig = max(0, min(x1_orig, orig_w))
                y1_orig = max(0, min(y1_orig, orig_h))
                x2_orig = max(0, min(x2_orig, orig_w))
                y2_orig = max(0, min(y2_orig, orig_h))

                # Draw bbox (green)
                cv2.rectangle(img_with_boxes,
                             (x1_orig, y1_orig), (x2_orig, y2_orig),
                             (0, 255, 0), 3)

                # Label background
                label = f"Class {cls_id}: {conf:.2f}"
                font_scale = 0.8
                thickness = 2
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                # Ensure label stays within image
                label_y = max(label_h + 10, y1_orig)

                cv2.rectangle(img_with_boxes,
                             (x1_orig, label_y - label_h - 10),
                             (x1_orig + label_w + 10, label_y),
                             (0, 255, 0), -1)

                # Label text
                cv2.putText(img_with_boxes, label,
                           (x1_orig + 5, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, (0, 0, 0), thickness)

    return img_with_boxes
