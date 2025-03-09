from enum import Enum

class YoloType():
    """Enumeration for model types."""

    class Pretrained(Enum):
        yolo8n = "yolo8n.pt"
        yolo8s = "yolo8s.pt"
        yolo8m = "yolo8m.pt"
        yolo8l = "yolo8l.pt"
        yolo8x = "yolo8x.pt"
        yolo9n = "yolo9n.pt"
        yolo9s = "yolo9s.pt"
        yolo9m = "yolo9m.pt"
        yolo9l = "yolo9l.pt"
        yolo9x = "yolo9x.pt"
        yolo10n = "yolo10n.pt"
        yolo10s = "yolo10s.pt"
        yolo10m = "yolo10m.pt"
        yolo10l = "yolo10l.pt"
        yolo10x = "yolo10x.pt"
        yolo11n = "yolo11n.pt"
        yolo11s = "yolo11s.pt"
        yolo11m = "yolo11m.pt"
        yolo11l = "yolo11l.pt"
        yolo11x = "yolo11x.pt"

    class Custom(Enum):
        Firearm_last = "models/last_Firearm.pt"
        Firearm_best = "models/best_Firearm.pt" 

    class sign(Enum):
        sign_model = "models/sign_model.pt"