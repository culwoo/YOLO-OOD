# 🚢 YOLO + OOD Detection Gradio Prototype

A web-based prototype for ship detection using YOLO combined with Mahalanobis distance-based Out-of-Distribution (OOD) detection.

## Features

- **YOLO Object Detection**: Detects ships in 3 categories (어선, 상선, 군함)
- **OOD Detection**: Identifies out-of-distribution samples using Mahalanobis distance
- **CLAHE Preprocessing**: Optional contrast enhancement for improved detection
- **Interactive Web Interface**: Built with Gradio for easy image upload and visualization
- **CPU-Only**: Optimized for local execution without GPU

## Project Structure

```
Final Gradio/
├── app.py                          # Gradio web interface
├── inference.py                    # YOLO + OOD detection logic
├── utils.py                        # Visualization and preprocessing utilities
├── requirements.txt                # Python dependencies
├── models/
│   ├── best.pt                     # YOLO model weights
│   └── id_stats.pkl                # OOD statistics (mu, cov_inv, threshold)
└── README.md                       # This file
```

## Installation

### 1. Clone or Navigate to Project Directory

```bash
cd "Final Gradio"
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# For CPU-only (recommended for laptops)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If you have CUDA-capable GPU:
```bash
pip install -r requirements.txt
```

## Usage

### Start the Gradio Server

```bash
python app.py
```

The server will start and display:
```
============================================================
Starting Gradio server...
Access the interface at: http://localhost:7860
Press Ctrl+C to stop the server
============================================================
```

### Using the Web Interface

1. Open your browser and go to `http://localhost:7860`
2. Upload an image using the image upload box
3. (Optional) Enable "Apply CLAHE Preprocessing" for contrast enhancement
4. Click the **Detect** button
5. View the results:
   - **Green boxes**: Detected ships with class labels
   - **Red border**: OOD warning (if detected)
   - **Score info**: Mahalanobis distance and threshold

### Stopping the Server

Press `Ctrl+C` in the terminal to stop the Gradio server.

## Understanding the Output

### Visualization Elements

- **Bounding Boxes (Green)**: Detected ships with class ID and confidence score
- **OOD Warning (Red Border)**: Appears when the image is classified as out-of-distribution
- **Score Information**:
  - **Score**: Mahalanobis distance from in-distribution mean
  - **Threshold**: Decision boundary for OOD classification
  - **Status**: "ID" (in-distribution) or "OOD" (out-of-distribution)

### CLAHE Preprocessing

CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances image contrast, which can improve:
- Detection in low-contrast conditions
- OOD detection sensitivity

Enable the checkbox to compare results with/without CLAHE.

## Model Details

### YOLO Model
- **Architecture**: YOLOv10-based
- **Classes**:
  - Class 0: 어선 (Fishing vessel)
  - Class 1: 상선 (Merchant ship)
  - Class 2: 군함 (Warship)
- **Input Size**: 960x544 (with LetterBox)

### OOD Detection
- **Method**: Mahalanobis distance on Neck layer features
- **Feature Extraction**: Global Average Pooling on layer 22
- **Threshold**: Loaded from `id_stats.pkl`

## Troubleshooting

### Port Already in Use

If port 7860 is already in use, edit `app.py` and change:
```python
demo.launch(server_port=7861)  # Use different port
```

### Out of Memory

If you encounter memory issues:
1. Close other applications
2. Ensure you're using CPU version of PyTorch
3. Process smaller images

### Slow Inference

CPU inference is slower than GPU. Typical processing time:
- Single image: 2-5 seconds (depending on image size)

To speed up:
- Resize large images before uploading
- Consider using GPU if available

## Development Notes

This prototype is based on Jupyter notebooks:
- `YOLO and OOD mahalanobis inference FPS.ipynb`: FPS benchmarking
- `인퍼런스최적화CLAHE및시각화추가.ipynb`: Visualization implementation

### Modifying the Code

- **Change model**: Replace `models/best.pt`
- **Adjust OOD threshold**: Modify `models/id_stats.pkl`
- **Customize visualization**: Edit `utils.py`
- **Add features**: Extend `inference.py` or `app.py`

## Requirements

- Python 3.8+
- 4GB+ RAM recommended
- No GPU required (CPU-only)

## License

This is a prototype for educational/research purposes.

## Contact

For issues or questions, please refer to the original notebook files or project documentation.
