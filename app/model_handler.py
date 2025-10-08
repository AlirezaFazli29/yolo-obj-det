from enum import Enum

class YoloType():
    """Enumeration for model types."""

    class Pretrained(Enum):
        yolo8n = "yolov8n.pt"
        yolo8s = "yolov8s.pt"
        yolo8m = "yolov8m.pt"
        yolo8l = "yolov8l.pt"
        yolo8x = "yolov8x.pt"
        yolo10n = "yolov10n.pt"
        yolo10s = "yolov10s.pt"
        yolo10m = "yolov10m.pt"
        yolo10l = "yolov10l.pt"
        yolo10x = "yolov10x.pt"
        yolo11n = "yolo11n.pt"
        yolo11s = "yolo11s.pt"
        yolo11m = "yolo11m.pt"
        yolo11l = "yolo11l.pt"
        yolo11x = "yolo11x.pt"
        yolo12n = "yolo12n.pt"
        yolo12s = "yolo12s.pt"
        yolo12m = "yolo12m.pt"
        yolo12l = "yolo12l.pt"
        yolo12x = "yolo12x.pt"

    class Custom(Enum):
        Firearm_last = "models/last_Firearm.pt"
        Firearm_best = "models/best_Firearm.pt" 