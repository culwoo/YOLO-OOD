# -*- coding: utf-8 -*-
"""
Gradio Web Interface for YOLO + OOD Detection
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

from inference import YOLOOODDetector


import threading
import traceback

# Global detector instance (loaded once at startup)
print("Initializing detector...")
detector = YOLOOODDetector(
    model_path="models/best.pt",
    stats_path="models/id_stats.pkl"
)
print("Detector ready!")

# Lock for thread safety (since detector uses hooks and shared state)
detector_lock = threading.Lock()


def process_image(image, use_clahe, yolo_conf_threshold, ood_threshold):
    """
    Process uploaded image with YOLO + OOD detection

    Args:
        image: Uploaded image (numpy array or PIL Image)
        use_clahe: Boolean, whether to apply CLAHE preprocessing
        yolo_conf_threshold: YOLO confidence threshold
        ood_threshold: OOD detection threshold

    Returns:
        result_image: Visualized result image
    """
    if image is None:
        return None

    try:
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Perform detection with lock
        with detector_lock:
            result_img, ood_score, is_ood, results = detector.detect(
                image,
                use_clahe=use_clahe,
                conf_threshold=yolo_conf_threshold,
                ood_threshold=ood_threshold
            )

        # Convert BGR to RGB for Gradio display
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        return result_img_rgb

    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        # Return the original image (or None) if detection fails, to avoid UI error
        # But we need to return something valid for gr.Image
        # Let's try to return the original image converted to numpy if possible
        try:
            if isinstance(image, Image.Image):
                return np.array(image)
            return image
        except:
            return None


# Build Gradio interface
with gr.Blocks(title="YOLO + OOD Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # YOLO 기반 선박 탐지 및 OOD 시스템
        
        이미지를 업로드하면 자동으로 선박 탐지 및 OOD 판정이 수행됩니다.
        우측의 슬라이더를 조절하여 실시간으로 결과를 확인할 수 있습니다.
        """
    )
    
    # State to hold the original uploaded image
    original_image_state = gr.State(value=None)
    # State to hold log history
    log_state = gr.State(value=[])

    def add_log(history, message):
        """Add a message to the log history with timestamp"""
        now = datetime.now().strftime("%H:%M:%S")
        new_log = f"[{now}] {message}"
        history.append(new_log)
        return history, "\n".join(history[::-1])  # Show newest first

    with gr.Row():
        with gr.Column(scale=1):
             # Help button: Small question mark icon
            help_btn = gr.Button("❓", scale=0, min_width=1)
            
    # Help area state
    help_visible_state = gr.State(value=False)

    with gr.Row(visible=False) as help_area:
        gr.Markdown(
            """
            ### 사용 가이드
            1. **이미지 입력**: 좌측의 이미지 영역을 클릭하여 이미지를 업로드하거나 클립보드에서 붙여넣으세요.
            2. **YOLO 임계값 설정**: 'YOLO 신뢰도 임계값' 슬라이더를 조절하여 탐지 민감도를 설정하세요.
            3. **OOD 임계값 설정**: 'OOD 임계값' 슬라이더를 조절하여 미상 물체 탐지 민감도를 설정하세요.
            """
        )

    with gr.Row():
        # Left Column: Image Display (Input & Output)
        with gr.Column(scale=2):
            image_display = gr.Image(
                type="pil",
                interactive=True,
                sources=["upload", "clipboard"],
                show_label=False
            )

        # Right Column: Controls
        with gr.Column(scale=1):
            gr.Markdown("### 설정")
            
            use_clahe_checkbox = gr.Checkbox(
                label="CLAHE 전처리 적용",
                value=False,
                info="대비 향상 활성화 (OOD 탐지에 도움될 수 있음)"
            )

            gr.Markdown("### 임계값 설정")

            yolo_conf_slider = gr.Slider(
                minimum=0.001,
                maximum=1.0,
                value=0.25,
                step=0.01,
                label="YOLO 신뢰도 임계값",
                info="값이 높을수록 고신뢰 탐지만 표시"
            )

            ood_threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=50.0,
                value=detector.threshold,
                step=0.1,
                label="OOD 임계값",
                info="값이 높을수록 OOD에 덜 민감 (경고 감소)"
            )

            gr.Markdown(
                """
                ### 범례
                - 초록 박스: 탐지된 선박 (어선, 상선, 군함)
                - 빨간 테두리: 미상 물체 경고 (OOD)
                - 점수: Mahalanobis 거리
                """
            )
            
            gr.Markdown("### 시스템 로그")
            log_box = gr.Textbox(
                label="Log",
                placeholder="시스템 상태 로그가 여기에 표시됩니다.",
                lines=5,
                interactive=False
            )

    # --- Event Handlers ---

    def on_upload(image, use_clahe, yolo_conf, ood_threshold, history):
        """
        Handle image upload:
        1. Save original image to state
        2. Run detection immediately
        3. Return original image (for state) and processed image (for display)
        """
        if image is None:
            return None, None, history, "\n".join(history[::-1])
        
        # Log upload
        history, log_text = add_log(history, "이미지가 업로드되었습니다.")
        
        # Run detection
        result_img = process_image(image, use_clahe, yolo_conf, ood_threshold)
        
        history, log_text = add_log(history, "탐지 완료.")
        
        # Return original image to state, and result to display
        return image, result_img, history, log_text

    def on_param_change(original_image, use_clahe, yolo_conf, ood_threshold, history):
        """
        Handle parameter change:
        1. Use original image from state
        2. Run detection
        3. Return processed image
        """
        if original_image is None:
            return None, history, "\n".join(history[::-1])
        
        # Check thresholds for warnings
        warnings = []
        if yolo_conf < 0.1:
            warnings.append("경고: YOLO 임계값이 너무 낮습니다 (오탐지 가능성).")
        elif yolo_conf > 0.9:
            warnings.append("경고: YOLO 임계값이 너무 높습니다 (미탐지 가능성).")
            
        if ood_threshold < 5.0:
            warnings.append("경고: OOD 임계값이 너무 낮습니다 (민감).")
        elif ood_threshold > 40.0:
            warnings.append("경고: OOD 임계값이 너무 높습니다 (둔감).")
            
        for w in warnings:
             history, _ = add_log(history, w)

        # Run detection on the stored original image
        result_img = process_image(original_image, use_clahe, yolo_conf, ood_threshold)
        
        history, log_text = add_log(history, f"설정 변경됨 (YOLO: {yolo_conf}, OOD: {ood_threshold})")
        
        return result_img, history, log_text

    def toggle_help(visible):
        new_visible = not visible
        return new_visible, gr.update(visible=new_visible)

    # 1. Image Upload Event
    image_display.upload(
        fn=on_upload,
        inputs=[image_display, use_clahe_checkbox, yolo_conf_slider, ood_threshold_slider, log_state],
        outputs=[original_image_state, image_display, log_state, log_box]
    )

    # 2. Parameter Change Events (Slider, Checkbox)
    # They use the State image, not the currently displayed image (which might already be annotated)
    params = [original_image_state, use_clahe_checkbox, yolo_conf_slider, ood_threshold_slider, log_state]
    
    use_clahe_checkbox.change(fn=on_param_change, inputs=params, outputs=[image_display, log_state, log_box])
    yolo_conf_slider.change(fn=on_param_change, inputs=params, outputs=[image_display, log_state, log_box])
    ood_threshold_slider.change(fn=on_param_change, inputs=params, outputs=[image_display, log_state, log_box])
    
    # Help Button Event
    help_btn.click(
        fn=toggle_help,
        inputs=[help_visible_state],
        outputs=[help_visible_state, help_area]
    )

    # 3. Clear Event
    def on_clear():
        return None
    
    image_display.clear(fn=on_clear, inputs=None, outputs=original_image_state)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Gradio server...")
    print("Access the interface at: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")

    demo.launch(
        server_name="0.0.0.0",  # Allow access from local network
        # server_port=7860,  # Removed to allow dynamic port allocation
        share=False,  # Local only (no public URL)
        show_error=True
    )
