from fastapi import (
    FastAPI,
    File, 
    UploadFile, 
    Form, 
    HTTPException,
)
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)
from model_handler import YoloType
from ultralytics import YOLO
from utils import (
    process_yolo_result,
    JSONRequest,
    JSONRequest2,
    ModelJSONRequest,
)
from contextlib import asynccontextmanager
from PIL import Image
import base64
import uvicorn
import io
import os


my_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.

    This function initializes and manages the lifecycle of YOLO models required 
    by the application. Models are loaded at the start of the application and 
    cleaned up upon shutdown.

    Args:
        app (FastAPI): The FastAPI application instance.

    Steps:
        1. Load a pretrained YOLO model for COCO dataset and store it in a global model dictionary.
        2. Load a custom YOLO firearm detection model from a specified path.
        3. Provide access to the models during the application's lifespan.
        4. Clear the models from memory during cleanup.

    Yields:
        None: Allows the application to use the models while the context is active.
    """
    yolo_coco = YOLO(YoloType.Pretrained.yolo11n.value)
    my_models["coco"] = yolo_coco
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gun_pth = os.path.join(base_dir, YoloType.Custom.Firearm_best.value)
    yolo_gun = YOLO(gun_pth)
    my_models["gun"] = yolo_gun
    yield
    my_models.clear()


app = FastAPI(
    title="object detection",
    lifespan=lifespan
)


@app.get(
    path="/", 
    tags=["Object Detection", "Firearm Classification", "Model Selection"]
)
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
    res = my_models["coco"](image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    return JSONResponse(response)


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
    res = my_models["gun"](image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    return JSONResponse(response)


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
    res = my_models["coco"](image, conf=conf_threshold, verbose=False)
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
    res = my_models["gun"](image, conf=conf_threshold, verbose=False)
    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/obj_process_n_return_result", tags=["Object Detection"])
async def obj_process_n_return_result(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
    return_base64_result: bool = Form,
):
    """
    Perform object detection on an uploaded image file with an option to return the result as a Base64-encoded image.

    Args:
        file (UploadFile): The uploaded image file to be processed for object detection.
        conf_threshold (float): The confidence threshold for the YOLO model to filter detections.
                                Values should typically be between 0 and 1.
        return_base64 (bool): Boolean flag indicating whether to return the result image as a Base64 string.

    Returns:
        JSONResponse or StreamingResponse: The processed YOLO inference result as a structured JSON response, 
                                           or as a JSON response containing the Base64-encoded image if requested.

    Raises:
        HTTPException: If the uploaded file is not a valid image or cannot be processed.
    """
    image = Image.open(file.file)
    res = my_models["coco"](image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["base64_result_image"] = base64_image
    return JSONResponse(response)


@app.post("/gun_process_n_return_result", tags=["Firearm Classification"])
async def gun_process_n_return_result(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
    return_base64_result: bool = Form,
):
    """
    Perform firearm classification on an uploaded image file.

    Args:
        file (UploadFile): The uploaded image file to be classified.
        conf_threshold (float): The confidence threshold for filtering model detections.
                                Must be between 0 and 1.
        return_base64_result (bool): A flag to indicate whether to return the result image as a Base64-encoded string.

    Returns:
        JSONResponse: A JSON object containing:
                      - Detection results (structured data from YOLO inference).
                      - Optionally, the result image encoded as a Base64 string, if `return_base64_result` is True.

    Raises:
        HTTPException: If the uploaded file is not valid or an error occurs during processing.
    """
    image = Image.open(file.file)
    res = my_models["gun"](image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["base64_result_image"] = base64_image
    return JSONResponse(response)


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
    res = my_models["coco"](image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    return JSONResponse(response)


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
    res = my_models["gun"](image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    return JSONResponse(response)


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
    res = my_models["coco"](image, conf=conf_threshold, verbose=False)
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
    res = my_models["gun"](image, conf=conf_threshold, verbose=False)
    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/obj_process_n_return_result_base64", tags=["Object Detection"])
async def obj_process_n_return_result_base64(request: JSONRequest2):
    """
    Perform object detection on an image provided as a Base64 string.

    Args:
        request (JSONRequest2): A JSON object containing:
            - base64_string (str): The Base64-encoded string of the input image.
            - conf_threshold (float): Confidence threshold for filtering detections.
            - return_base64_result (bool): Whether to include the result image as a Base64-encoded string in the response.

    Returns:
        JSONResponse: A JSON object containing:
            - Detection results from YOLO inference.
            - Optionally, the result image encoded as a Base64 string if `return_base64_result` is True.

    Raises:
        HTTPException: If there are issues with the input or processing the image.
    """
    image_data = base64.b64decode(request.base64_string)
    image = Image.open(io.BytesIO(image_data))
    conf_threshold = request.conf_threshold
    res = my_models["coco"](image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    return_base64_result = request.return_base64_result
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["base64_result_image"] = base64_image
    return JSONResponse(response)


@app.post("/gun_process_n_return_result_base64", tags=["Firearm Classification"])
async def gun_process_n_return_result_base64(request: JSONRequest2):
    """
    Perform firearm classification on an image provided as a Base64 string.

    Args:
        request (JSONRequest2): A JSON object containing:
            - base64_string (str): The Base64-encoded string of the input image.
            - conf_threshold (float): Confidence threshold for filtering detections.
            - return_base64_result (bool): Whether to include the result image as a Base64-encoded string in the response.

    Returns:
        JSONResponse: A JSON object containing:
            - Detection results from YOLO inference.
            - Optionally, the result image encoded as a Base64 string if `return_base64_result` is True.

    Raises:
        HTTPException: If there are issues with the input or processing the image.
    """
    image_data = base64.b64decode(request.base64_string)
    image = Image.open(io.BytesIO(image_data))
    conf_threshold = request.conf_threshold
    res = my_models["gun"](image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    return_base64_result = request.return_base64_result
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["base64_result_image"] = base64_image
    return JSONResponse(response)


@app.post("/select_coco_model", tags=["Model Selection"])
async def select_coco(
    model_type: YoloType.Pretrained = Form(...)
):
    """
    Endpoint to select and load a COCO model.

    This endpoint allows the user to dynamically select a COCO model 
    from the predefined YOLO pretrained models.

    Args:
        model_type (YoloType.Pretrained): The type of COCO model to load, 
                                          provided as a form input.

    Returns:
        JSONResponse: A JSON response confirming the selected model.
    """
    global my_models
    my_models["coco"] = YOLO(model_type.value)
    return JSONResponse(
        {"message": f"Model {model_type.name} is selected"}
    )


@app.post("/select_gun_model", tags=["Model Selection"])
async def select_gun(
    model_type: YoloType.Custom = Form(...)
):
    """
    Endpoint to select and load a firearm classification model.

    This endpoint allows the user to dynamically select a custom firearm 
    classification model. It loads the model based on the provided configuration.

    Args:
        model_type (YoloType.Custom): The type of firearm classification model to
                                      load, provided as a form input.

    Returns:
        JSONResponse: A JSON response confirming the selected model.
    """
    global my_models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gun_pth = os.path.join(base_dir, YoloType.Custom.Firearm_best.value)
    my_models["gun"] = YOLO(gun_pth)
    return JSONResponse(
        {"message": f"Model {model_type.name} is selected"}
    )


@app.get("/show_model_types", tags=["Model Selection"])
async def show_model_types():
    coco_types = {
        model_type.name: model_type.value for model_type in YoloType.Pretrained
    }
    gun_type =  {
        model_type.name: model_type.value for model_type in YoloType.Custom
    }
    return JSONResponse(
        {
            "Object Detection Models": coco_types,
            "Firearm Classification Models": gun_type
        }
    )


@app.post("/select_coco_model_json_request", tags=["Model Selection"])
async def select_coco_model_json_request(
    request: ModelJSONRequest
):
    """
    Endpoint to select and load a COCO model based on a JSON request.

    This endpoint allows the user to dynamically select a COCO model 
    by sending the model type in a JSON payload.

    Args:
        request (ModelJSONRequest): The JSON request body containing the model type to load.

    Returns:
        JSONResponse: A JSON response confirming the selected model.
    """
    global my_models
    my_models["coco"] = YOLO(request.model_type)
    return JSONResponse(
        {"message": f"Model from path {request.model_type} is selected"}
    )


@app.post("/select_gun_model_json_request", tags=["Model Selection"])
async def select_gun_model_json_request(
    request: ModelJSONRequest
):
    """
    Endpoint to select and load a firearm classification model based on a JSON request.

    This endpoint enables dynamic selection of a firearm classification model 
    by specifying the model type in a JSON payload.

    Args:
        request (ModelJSONRequest): The JSON request body containing the model type to load.

    Returns:
        JSONResponse: A JSON response confirming the selected model.
    """
    global my_models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gun_pth = os.path.join(base_dir, request.model_type)
    my_models["gun"] = YOLO(gun_pth)
    return JSONResponse(
        {"message": f"Model from path {request.model_type} is selected"}
    )


uvicorn.run(app, host="0.0.0.0", port=8080)