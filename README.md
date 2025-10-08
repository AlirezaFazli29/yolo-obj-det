# Object Detection & Firearm Classification API

This project provides a **FastAPI-based service** for object detection and firearm classification using **YOLO models**. The API supports both image file uploads and Base64-encoded images, and can return detection results as JSON or visualized images.

> ⚠️ Note: Firearm detection models are pretrained and cannot be included in this repository.

---

## Features

- Object detection using YOLO pretrained models (COCO)  
- Firearm detection using custom YOLO models  
- Supports file uploads and Base64 images  
- Returns results as:
  - Structured JSON (with bounding boxes, classes, counts)  
  - Visualized images (JPEG streaming response)  
  - Base64-encoded images (full or cropped detections)  
- Async model loading for improved performance  

---

## Installation

### 1. Python Environment

```bash
git clone git@github.com:AlirezaFazli29/yolo-obj-det.git
cd <repo_folder>
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Docker (Optional)

Build the Docker image:

```bash
docker compose build
```


Run the container with GPU support (requires NVIDIA runtime):

```bash
docker run --gpus all -p 8080:8080 object-detection-app
```

Also you can run with the compose file:
```bash
docker compose up -d
```

> Make sure your Docker daemon supports NVIDIA runtime. Example `daemon.json`:

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

---

## Running the Application

You can run the app directly with Python:

```bash
python -m app.main
```

The service will be available at:

```
http://localhost:8080/
```

---

## API Endpoints

- `GET /` – Health check  
- `POST /file-to-base64` – Convert uploaded file to Base64  
- `POST /obj_process` – Object detection on uploaded file  
- `POST /gun_process` – Firearm detection on uploaded file  
- `POST /obj_process_plot` – Object detection with visualized result  
- `POST /gun_process_plot` – Firearm detection with visualized result  
- Base64 versions of all above endpoints are also available:
  - `/obj_process_base64`
  - `/gun_process_base64`
  - `/obj_process_plot_base64`
  - `/gun_process_plot_base64`
- Dynamic model selection:
  - `/select_coco_model` – Select COCO model via form  
  - `/select_gun_model` – Select firearm model via form  
  - JSON request alternatives are also available.

---

## Model Requirements

- **COCO models**: Pretrained YOLO weights included (`yolo11n.pt`, `yolo11s.pt`, etc.)  
- **Firearm models**: Pretrained models (`best_Firearm.pt`, `last_Firearm.pt`) **must be downloaded separately** due to licensing restrictions.

Place firearm models in:

```
app/models/
```

---

## Notes

- API uses **async context manager** for YOLO model loading to improve performance and reduce memory usage.  
- Designed to support GPU acceleration with **NVIDIA runtime** in Docker.  
- JSON and Base64 endpoints allow flexibility for web or mobile client integration.

## License

MIT License

Copyright (c) 2025 Alireza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---
