from fastapi import (
        FastAPI, 
        Request, 
        File, 
        UploadFile, 
        Form, 
        HTTPException
    )
from fastapi.responses import (
        JSONResponse,
        StreamingResponse                    
    )
from model_handler import YoloType
from ultralytics import YOLO
from utils import (
    process_yolo_result,
    JSONRequest
)
from PIL import Image
import base64
import uvicorn
import io
import os


app = FastAPI(title="object detection")

yolo_coco = YOLO(YoloType.Pretrained.yolo11n.value)
base_dir = os.path.dirname(os.path.abspath(__file__))
gun_pth = os.path.join(base_dir, YoloType.Custom.Firearm_best.value)
yolo_gun = YOLO(gun_pth)


@app.get("/", tags=["Object Detection", "Firearm Classification"])
async def root():
    """
    Root endpoint for the service.

    This endpoint serves as a basic health check to verify if the service is up and running.
    When accessed, it returns a simple message indicating that the service is operational.

    Returns:
        JSONResponse: A JSON response with a message confirming the service status.
    """
    return JSONResponse(
        {"message": "Service is up and running"}
    )


@app.post("/file-to-base64", tags=["Object Detection", "Firearm Classification"])
async def file_to_base64(file: UploadFile):
    """
    Convert an uploaded file to a base64-encoded string.

    This function processes an uploaded file and converts its binary content
    into a base64-encoded string. The resulting string can be used for
    serialization or transmission of the file content in text format.
    If any error occurs during file processing, it raises an appropriate 
    HTTP exception with a message.

    Args:
        file (UploadFile): The file uploaded by the client.

    Returns:
        JSONResponse: A JSON response containing the filename and the 
                      base64-encoded string of the file content.
    """
    try:
        file_data = await file.read()
        base64_string = base64.b64encode(file_data).decode('utf-8')
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process the uploaded file.")

    return JSONResponse(
        content={"filename": file.filename, "base64_string": base64_string}
    )


@app.post("/obj_process", tags=["Object Detection"])
async def obj_process(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    """
    Perform object detection on an uploaded image file.

    This function processes an uploaded image file using a YOLO object detection model
    and applies the specified confidence threshold for detection. It returns the processed 
    result in a structured format.

    Args:
        file (UploadFile): The uploaded image file to be processed for object detection.
        conf_threshold (float): The confidence threshold for the YOLO model to filter detections.
                                Values should typically be between 0 and 1.

    Returns:
        JSONResponse: The processed YOLO inference result as a structured JSON response.

    Raises:
        HTTPException: If the uploaded file is not a valid image or cannot be processed.
    """
    image = Image.open(file.file)
    res = yolo_coco(image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    return response


@app.post("/gun_process", tags=["Firearm Classification"])
async def gun_process(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    """
    Perform firearm classification on an uploaded image file.

    This function processes an uploaded image file using a YOLO firearm classification model
    and applies the specified confidence threshold for detection. It returns the processed
    result in a structured format.

    Args:
        file (UploadFile): The uploaded image file to be analyzed for firearm detection.
        conf_threshold (float): The confidence threshold for the YOLO model to filter detections.
                                Values should typically be between 0 and 1.

    Returns:
        JSONResponse: The processed YOLO inference result for firearm classification 
                      as a structured JSON response.

    Raises:
        HTTPException: If the uploaded file is not a valid image or cannot be processed.
    """
    image = Image.open(file.file)
    res = yolo_gun(image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    return response


@app.post("/obj_process_plot", tags=["Object Detection"])
async def obj_process_plot(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    """
    Perform object detection on an uploaded image and return a visualized result.

    This endpoint processes an uploaded image using a YOLO object detection model,
    applies the specified confidence threshold, and returns the visualized detection 
    result as a streaming response (image).

    Args:
        file (UploadFile): The uploaded image file to be processed.
        conf_threshold (float): The confidence threshold for filtering detections.

    Returns:
        StreamingResponse: An image response containing the visualized detection result.
    
    Raises:
        HTTPException: If the uploaded file is not a valid image or cannot be processed.
    """
    image = Image.open(file.file)
    res = yolo_coco(image, conf=conf_threshold, verbose=False)
    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/gun_process_plot", tags=["Firearm Classification"])
async def gun_process_plot(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    """
    Perform firearm detection on an uploaded image and return a visualized result.

    This endpoint processes an uploaded image using a YOLO firearm classification model,
    applies the specified confidence threshold, and returns the visualized detection
    result as a streaming image response.

    Args:
        file (UploadFile): The uploaded image file to be analyzed for firearm detection.
        conf_threshold (float): The confidence threshold for filtering detections.
                                Values should typically range between 0 and 1.

    Returns:
        StreamingResponse: An image response containing the visualized firearm detection result.

    Raises:
        HTTPException: If the uploaded file is not a valid image or cannot be processed.
    """
    image = Image.open(file.file)
    res = yolo_gun(image, conf=conf_threshold, verbose=False)
    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/obj_process_base64", tags=["Object Detection"])
async def obj_process_base64(request: JSONRequest):
    """
    Process an object detection request using a base64-encoded image.

    This function accepts a JSON input containing a base64-encoded image and a 
    confidence threshold. It decodes the image, performs object detection using 
    the YOLO model, and processes the results into a structured JSON response.

    Args:
        request (JSONRequest): A JSON payload containing:
            - base64_string (str): The base64-encoded image data.
            - conf_threshold (float): The confidence threshold for object detection.

    Returns:
        JSONResponse: A JSON response with the processed object detection results. 
                      The response structure contains object types, counts, and 
                      bounding box data.

    Raises:
        HTTPException: If the base64 string is invalid, the image cannot be processed, 
                       or required fields are missing in the request.
    """
    image_data = base64.b64decode(request.base64_string)
    image = Image.open(io.BytesIO(image_data))
    conf_threshold = request.conf_threshold
    res = yolo_coco(image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    return response


@app.post("/gun_process_base64", tags=["Firearm Classification"])
async def gun_process_base64(request: JSONRequest):
    """
    Process a firearm classification request using a base64-encoded image.

    This function accepts a JSON input containing a base64-encoded image and a 
    confidence threshold. It decodes the image, performs firearm classification 
    using the YOLO model, and processes the results into a structured JSON response.

    Args:
        request (JSONRequest): A JSON payload containing:
            - base64_string (str): The base64-encoded image data to be analyzed.
            - conf_threshold (float): The confidence threshold for firearm classification.
                                      Values should typically range between 0 and 1.

    Returns:
        JSONResponse: A JSON response with the processed firearm classification results. 
                      The response includes detected firearm types, counts, and bounding 
                      box data.

    Raises:
        HTTPException: If the base64 string is invalid, the image cannot be processed, 
                       or required fields are missing in the request.
    """
    image_data = base64.b64decode(request.base64_string)
    image = Image.open(io.BytesIO(image_data))
    conf_threshold = request.conf_threshold
    res = yolo_gun(image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    return response


@app.post("/obj_process_plot_base64", tags=["Object Detection"])
async def obj_process_plot_base64(request: JSONRequest):
    """
    Process an object detection request and return a plotted image as a PNG.

    This endpoint accepts a JSON input containing a base64-encoded image and a 
    confidence threshold. It decodes the image, performs object detection using 
    the YOLO model, plots the detection results on the image, and returns the 
    plotted image as a streaming response in PNG format.

    Args:
        request (JSONRequest): A JSON payload containing:
            - base64_string (str): The base64-encoded image data.
            - conf_threshold (float): The confidence threshold for object detection.

    Returns:
        StreamingResponse: A streaming response containing the plotted image in 
                           PNG format. The image includes bounding boxes and 
                           annotations for detected objects.
        
    Raises:
        HTTPException: If the base64 string is invalid, the image cannot be processed, 
                       or required fields are missing in the request.
    """
    image_data = base64.b64decode(request.base64_string)
    image = Image.open(io.BytesIO(image_data))
    conf_threshold = request.conf_threshold
    res = yolo_coco(image, conf=conf_threshold, verbose=False)
    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/gun_process_plot_base64", tags=["Firearm Classification"])
async def gun_process_plot_base64(request: JSONRequest):
    """
    Perform firearm classification on a base64-encoded image and return a visualized result.

    This function accepts a JSON input containing a base64-encoded image and a confidence 
    threshold. It decodes the image, performs firearm classification using the YOLO model, 
    and returns the visualized detection result as a PNG image in a streaming response.

    Args:
        request (JSONRequest): A JSON payload containing:
            - base64_string (str): The base64-encoded image data to be analyzed.
            - conf_threshold (float): The confidence threshold for firearm detection.

    Returns:
        StreamingResponse: A response containing the visualized firearm detection result 
                           as a PNG image.

    Raises:
        HTTPException: If the base64 string is invalid, the image cannot be processed, 
                       or required fields are missing in the request.
    """
    image_data = base64.b64decode(request.base64_string)
    image = Image.open(io.BytesIO(image_data))
    conf_threshold = request.conf_threshold
    res = yolo_gun(image, conf=conf_threshold, verbose=False)
    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


uvicorn.run(app, host="0.0.0.0", port=8080)