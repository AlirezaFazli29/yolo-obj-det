from ultralytics.engine.results import Results
from pydantic import BaseModel
from base64 import b64encode
from numpy import array
from io import BytesIO
from PIL import Image



def process_yolo_result(
        result: Results,
        return_base64_cropped_objects: bool=False,
) -> list:
    """
    Process the YOLO inference result and group detections by object type.

    This function takes the result from a YOLO inference, which includes detected 
    objects with their respective bounding boxes and confidence scores, and then 
    groups these detections by object name (e.g., "person", "car", etc.). It returns 
    a list of dictionaries containing the object name, the count of detections for each object, 
    and the associated bounding boxes and confidence scores.

    Args:
        result (Results): The result from the YOLO model inference, typically containing 
                          a list of detected objects with their properties (name, 
                          confidence, bounding box coordinates).

    Returns:
        list: A list of dictionaries where each dictionary contains:
              - 'obj': The object name.
              - 'count': The number of detections for this object.
              - 'boxes': A list of bounding box data for each detection, including 
                confidence score and coordinates (x1, y1, x2, y2).
              If no objects are detected, a dictionary with an error message is returned.
    """
    summary = result.summary()
    if return_base64_cropped_objects:
        origin_img = result.orig_img[:, :, ::-1]
    if len(summary) > 0:
        grouped = {}
        for item in summary:
            obj_name = item['name']
            if obj_name not in grouped:
                grouped[obj_name] = {'obj': obj_name, 'count': 0, 'boxes': []}
            box_info = {
                'conf': item['confidence'],
                **item['box'],
            }
            grouped[obj_name]['boxes'].append(box_info)
            boxes = grouped[obj_name]['boxes']
            if return_base64_cropped_objects:
                for i, box in enumerate(boxes):
                    x1, y1 = int(box['x1']), int(box['y1'])
                    x2, y2 = int(box['x2']), int(box['y2'])
                    temp_img = array(origin_img)[y1:y2, x1:x2, :]
                    temp_img = Image.fromarray(temp_img)
                    buffer = BytesIO()
                    temp_img.save(buffer, format="JPEG")
                    buffer.seek(0)
                    temp_img = b64encode(buffer.getvalue()).decode('utf-8')
                    grouped[obj_name]['boxes'][i]["base64_cropped"] = temp_img
            grouped[obj_name]['count'] += 1
        return list(grouped.values())
    else: 
        return [[]]


class JSONRequest(BaseModel):
    base64_string: str
    conf_threshold: float


class JSONRequest2(BaseModel):
    base64_string: str
    conf_threshold: float = 0.5
    return_base64_result: bool = True
    return_base64_cropped_objects: bool = True


class ModelJSONRequest(BaseModel):
    model_type: str