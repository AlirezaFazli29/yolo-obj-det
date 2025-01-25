from ultralytics.engine.results import Results
from pydantic import BaseModel


def process_yolo_result(result: Results) -> dict:
    """
    Process the YOLO inference result and group detections by object type.

    This function takes the result from a YOLO inference, which includes detected 
    objects with their respective bounding boxes and confidence scores, and then 
    groups these detections by object name (e.g., "person", "car", etc.). It returns 
    a dictionary containing the object name, the count of detections for each object, 
    and the associated bounding boxes and confidence scores.

    Args:
        result (Results): The result from the YOLO model inference, typically containing 
                           a list of detected objects with their properties (name, 
                           confidence, bounding box coordinates).

    Returns:
        dict: A dictionary where the keys are object names (e.g., "person"), and the 
              values are dictionaries containing:
              - 'obj': The object name.
              - 'count': The number of detections for this object.
              - 'boxes': A list of bounding box data for each detection, including 
                confidence score and coordinates (x1, y1, x2, y2).
    """
    summary = result.summary()
    if len(summary)>0:
        grouped = {}
        for item in summary:
            obj_name = item['name']
            if obj_name not in grouped:
                grouped[obj_name] = {'obj_cls': obj_name, 'count': 0, 'boxes': []}
            grouped[obj_name]['boxes'].append({'conf': item['confidence'], **item['box']})
            grouped[obj_name]['count'] += 1
        return grouped
    else: 
        return {
            "error": 404,
            "message": "No object detected"
        }


class JSONRequest(BaseModel):
    base64_string: str
    conf_threshold: float