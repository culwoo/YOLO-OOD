"""
Utility functions for YOLO + OOD detection visualization
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def get_font(font_size):
    """Helper to load font"""
    try:
        return ImageFont.truetype("malgun.ttf", font_size)
    except:
        return ImageFont.load_default()

def get_text_size_korean(text, font_size):
    """
    Get text size and offsets using PIL
    """
    font = get_font(font_size)
    bbox = font.getbbox(text)
    # bbox is (left, top, right, bottom)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    offset_x = bbox[0]
    offset_y = bbox[1]
    return width, height, offset_x, offset_y

def draw_text_korean(img, text, position, color=(0, 255, 0), font_size=20):
    """
    Draw Korean text using PIL
    
    Args:
        img: (H, W, 3) BGR image (OpenCV format)
        text: Text to draw
        position: (x, y) coordinates
        color: (B, G, R) color tuple
        font_size: Font size
        
    Returns:
        img_with_text: (H, W, 3) BGR image
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font = get_font(font_size)
    
    # Convert BGR color to RGB for PIL
    color_rgb = (color[2], color[1], color[0])
    
    draw.text(position, text, font=font, fill=color_rgb)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


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
    text = "WARNING: 미상 물체 감지됨"
    
    # ------------------
    # (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    #x = (W - text_w) // 2
    #y = int(H * 0.08)  # Top 8% position

    # Text background (black rectangle)
    #padding = 15
    #cv2.rectangle(img_with_warning,
    #              (x - padding, y - text_h - padding),
    #              (x + text_w + padding, y + padding),
    #              (0, 0, 0), -1)
    # ------------------

    # Text (red in BGR) - Use PIL for Korean support
    # Re-calculate position and size using PIL font
    font_size = int(font_scale * 30) # Approximate conversion
    text_w, text_h, off_x, off_y = get_text_size_korean(text, font_size)

    x = (W - text_w) // 2
    y = int(H * 0.08)
    
    # Draw background again with correct size
    # Add some padding
    padding_x = 10
    padding_y = 10 # Increased vertical padding slightly
    
    cv2.rectangle(img_with_warning,
                  (x - padding_x, y - padding_y),
                  (x + text_w + padding_x, y + text_h + padding_y),
                  (0, 0, 0), -1)
                  
    # Draw text centered in the box
    # To align visible pixels to (x, y), we subtract the offsets
    draw_x = x - off_x
    draw_y = y - off_y
    
    img_with_warning = draw_text_korean(img_with_warning, text, (draw_x, draw_y), (0, 0, 255), font_size)

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

                # Class name mapping
                class_names = {
                    0: "어선",
                    1: "상선",
                    2: "군함"
                }
                class_name = class_names.get(cls_id, f"Class {cls_id}")
                
                # Label background
                label = f"{class_name}: {conf:.2f}"
                font_size = 20
                label_w, label_h, off_x, off_y = get_text_size_korean(label, font_size)
                
                # Ensure label stays within image
                label_y = max(label_h + 10, y1_orig)
                
                # Draw green background for label
                padding = 4
                # Box top is label_y - label_h - 2*padding
                # Box bottom is label_y
                box_top = label_y - label_h - padding * 2
                box_bottom = label_y
                box_left = x1_orig
                box_right = x1_orig + label_w + padding * 2
                
                cv2.rectangle(img_with_boxes,
                             (box_left, box_top),
                             (box_right, box_bottom),
                             (0, 255, 0), -1)

                # Label text
                # We want visible text top at box_top + padding
                # visible text top = draw_y + off_y
                # draw_y + off_y = box_top + padding
                # draw_y = box_top + padding - off_y
                
                text_draw_y = box_top + padding - off_y
                text_draw_x = box_left + padding - off_x
                
                img_with_boxes = draw_text_korean(
                    img_with_boxes, 
                    label, 
                    (text_draw_x, text_draw_y), 
                    (0, 0, 0), 
                    font_size
                )

    return img_with_boxes
