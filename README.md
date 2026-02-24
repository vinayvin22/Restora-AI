# RestoraAI: Supervised Heritage Image Restoration

RestoraAI is an advanced image restoration project designed to reconstruct and enhance degraded images of historical monuments. It utilizes a supervised U-Net architecture with a dilated bottleneck to handle noise, scratches, and missing patches while preserving structural fidelity and high clarity.

## Features
- **Supervised Restoration**: Trained on clean/distorted pairs for precise reconstruction.
- **Inpainting Capability**: Dilated convolutions in the bottleneck allow for effective reconstruction of missing regions.
- **High-Fidelity Output**: Prioritizes L1, SSIM, and Perceptual losses to ensure sharp and colorful results.
- **Demo-Ready Interface**: Simple one-click demonstration to visualize the restoration pipeline.

## Project Structure
- `backend/`: FastAPI server for handling restoration requests.
- `frontend/`: Modern web interface for demonstrating results.
- `models/`: PyTorch implementation of the Generator (U-Net).
- `data_prep/`: Scripts for image distortion and gallery generation.
- `training/`: Supervised training pipelines.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- FastAPI & Uvicorn

### Running the Demo
1. **Model Weights**: Ensure `saved_models/restora_gen.pth` is present.
2. **Start Backend & Frontend**:
   ```bash
   python run_app.py
   ```
3. **Access**: Open `http://127.0.0.1:8000` in your browser.

## Authors
- Vinay (vinayvin22)
