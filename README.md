# Drone Catcher

Drone detection and segmentation using PyTorch and Mask R-CNN, served through a FastAPI backend and a Streamlit interface.

This repository contains the code needed to run inference. It was adapted from [SolarPanel-Damage-Detector](https://github.com/Ptzatzag/SolarPanel-Damage-Detector).

## Features

- Upload JPG, JPEG, PNG images, or mp4 videos.
- Detect drones and display bounding boxes, confidence scores, and segmentation masks.
- Inspect predictions as JSON.
- Run the backend and frontend with Docker Compose.

## Repository structure

```text
Deployment/
├── app/                    # FastAPI backend and inference
│   ├── shared/             # Image preprocessing
│   ├── Dockerfile
│   ├── inference.py
│   ├── main.py
│   └── requirements.txt
├── frontend/               # Streamlit interface
│   ├── Dockerfile
│   ├── SLapp.py
│   └── requirements.txt
├── docker-compose.yml      # Build and run locally
model/
└── maskrcnn.py             # Model architecture and checkpoint loading
LICENSE
README.md
```

## Quick start

Clone the repository:

```bash
git clone https://github.com/Ptzatzag/drone_catcher.git
cd drone_catcher
```

Build and start the services:

```bash
docker compose -f Deployment/docker-compose.yml up --build -d
```

### Predict

Upload an image or a video to detect drones:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/drone.jpg"
```

The response contains a `predictions` list. Each detection includes:

| Field | Description |
| --- | --- |
| `label` | Numeric class ID |
| `score` | Detection confidence |
| `box` | Bounding box coordinates: `[x1, y1, x2, y2]` |
| `mask` | Binary segmentation mask |

Only detections with confidence scores above `0.7` are returned.

## Model weights

The backend loads its checkpoint from Hugging Face:

- Repository: `Ptzatzag/drone-catcher`
- File: `drone_weights.pth/Best_Model.pth`

Weights are downloaded at runtime and are not included in this repository.

The model configuration must match the checkpoint. The current implementation uses two classes, including the background class.


## Troubleshooting

- **The frontend cannot connect:** Check the backend logs and wait for model loading to finish.
- **The model download fails:** Check internet access, the Hugging Face checkpoint path, and any repository access requirements.
- **No drones are detected:** Predictions below the confidence threshold are filtered out.
- **A port is already in use:** Change the host port mapping in the Compose file.

## Video prediction example

Drone detection on a sample video using Mask R-CNN.

https://github.com/user-attachments/assets/1e99f11f-87e6-46de-80b1-4bf211340fef

## License

MIT License. See [LICENSE](LICENSE).
