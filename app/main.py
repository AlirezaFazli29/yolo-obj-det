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
from .model_handler import YoloType
from ultralytics import YOLO
from .utils import (
    process_yolo_result,
    JSONRequest,
    JSONRequest2,
    ModelJSONRequest,
)
from contextlib import asynccontextmanager
from torch.cuda import empty_cache
from PIL import Image
import asyncio
import uvicorn
import base64
import io
import os
import gc


my_models = {}
model_locks = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.

    This function initializes and manages the lifecycle of YOLO models used by the app.
    It loads the required models once at startup, creates dedicated asyncio locks for
    each model to ensure thread-safe access, and properly cleans up on shutdown.

    Steps:
        1. Load the pretrained YOLO model for general object detection (COCO dataset).
        2. Load the custom YOLO model for firearm detection from the specified path.
        3. Create a unique asyncio.Lock for each model to prevent concurrent inference calls
           on the same model instance.
        4. Store the models and locks globally for use across API endpoints.
        5. Clear all loaded models and locks on application shutdown.

    This design ensures that multiple requests can safely run concurrently using
    different models while avoiding race conditions or resource conflicts on the same model.

    Yields:
        None: Allows the application to use the models while the context is active.
    """
    yolo_coco = YOLO(YoloType.Pretrained.yolo11n.value)
    my_models["coco"] = yolo_coco
    model_locks["coco"] = asyncio.Lock()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    gun_pth = os.path.join(base_dir, YoloType.Custom.Firearm_best.value)
    yolo_gun = YOLO(gun_pth)
    my_models["gun"] = yolo_gun
    model_locks["gun"] = asyncio.Lock()

    yield

    for model in my_models.values():
        del model
    gc.collect()
    empty_cache()
    my_models.clear()
    model_locks.clear()



app = FastAPI(
    title="object detection",
    lifespan=lifespan
)


@app.get(
    path="/", 
    tags=[
        "Object Detection",
        "Firearm Classification",
        "Model Selection",
    ]
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
        {
            "message": "Service is up and running",
            "models": [
                "Object Detection",
                "Firearm Classification",
            ]
        }
    )


@app.post(
    "/file-to-base64",
    tags=[
        "Object Detection",
        "Firearm Classification",
    ]
)
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
    Perform object detection on an uploaded image file using the YOLO COCO model.

    This endpoint safely runs inference in an asynchronous environment by locking
    access to the shared YOLO model and offloading the blocking inference call
    to a background thread using `asyncio.to_thread`.

    Args:
        file (UploadFile): The uploaded image to be processed for object detection.
        conf_threshold (float): Confidence threshold for detection filtering (0-1).

    Returns:
        JSONResponse: Structured YOLO inference results containing detected objects.

    Concurrency:
        - Each model instance is protected by an `asyncio.Lock` to prevent simultaneous
          inference calls on the same model, which could cause memory corruption or crashes.
        - The actual YOLO inference runs in a separate background thread, ensuring that
          the FastAPI event loop remains responsive during computation.

    Raises:
        HTTPException: If the uploaded file is not a valid image.
    """
    try:
        image = Image.open(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    async with model_locks["coco"]:
        res = await asyncio.to_thread(
            my_models["coco"], image, conf=conf_threshold, verbose=False
        )
    
    response = process_yolo_result(res[0])
    return JSONResponse(response)


@app.post("/gun_process", tags=["Firearm Classification"])
async def gun_process(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    """
    Perform firearm classification on an uploaded image file using the YOLO firearm model.

    This endpoint handles firearm detection requests asynchronously and safely, even under 
    concurrent access. A per-model lock ensures that only one inference runs on the same YOLO 
    model instance at a time, preventing GPU memory conflicts and race conditions.

    The image is processed in a separate thread using `asyncio.to_thread()` to avoid blocking 
    the FastAPI event loop. Other requests targeting the same model will wait until the current 
    inference completes, while requests to other models can proceed in parallel.

    Args:
        file (UploadFile): The uploaded image file to analyze for firearm detection.
        conf_threshold (float): The confidence threshold for the YOLO model to filter detections.
                                Values should typically range between 0 and 1.

    Returns:
        JSONResponse: A structured JSON response containing the processed YOLO firearm detection results.

    Raises:
        HTTPException: If the uploaded file is not a valid image or cannot be processed.
    """
    try:
        image = Image.open(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    async with model_locks["gun"]:
        res = await asyncio.to_thread(
            my_models["gun"], image, conf=conf_threshold, verbose=False
        )
    
    response = process_yolo_result(res[0])
    return JSONResponse(response)


@app.post("/obj_process_plot", tags=["Object Detection"])
async def obj_process_plot(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    """
    Perform object detection on an uploaded image and return a visualized result.

    This endpoint processes an uploaded image using the YOLO COCO object detection model.
    It applies the specified confidence threshold to filter detections and generates a
    visualized output image with bounding boxes and labels. The image is returned as a
    streaming JPEG response.

    Args:
        file (UploadFile): The uploaded image file to be processed.
        conf_threshold (float): Confidence threshold (0-1) to filter detections.

    Returns:
        StreamingResponse: JPEG image containing the visualized detection results.

    Raises:
        HTTPException: If the uploaded file is invalid or cannot be processed.
    """
    try:
        image = Image.open(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    async with model_locks["coco"]:
        res = await asyncio.to_thread(
            my_models["coco"], image, conf=conf_threshold, verbose=False
        )

    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")


@app.post("/gun_process_plot", tags=["Firearm Classification"])
async def gun_process_plot(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    """
    Perform firearm detection on an uploaded image and return a visualized result.

    This endpoint processes an uploaded image using the YOLO firearm classification model.
    It applies the specified confidence threshold to filter detections and generates a
    visualized output image with bounding boxes and labels highlighting detected firearms.
    The image is returned as a streaming JPEG response.

    Args:
        file (UploadFile): The uploaded image file to be analyzed for firearm detection.
        conf_threshold (float): Confidence threshold (0-1) to filter detections.

    Returns:
        StreamingResponse: JPEG image containing the visualized firearm detection results.

    Raises:
        HTTPException: If the uploaded file is invalid or cannot be processed.
    """
    try:
        image = Image.open(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    async with model_locks["gun"]:
        res = await asyncio.to_thread(
            my_models["gun"], image, conf=conf_threshold, verbose=False
        )
    
    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")


@app.post("/obj_process_n_return_result", tags=["Object Detection"])
async def obj_process_n_return_result(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
    return_base64_result: bool = Form(...),
    return_base64_cropped_objects: bool = Form(...),
):
    """
    Perform object detection on an uploaded image and optionally return results as Base64.

    This endpoint processes an uploaded image using the YOLO COCO object detection model.
    It applies the specified confidence threshold to filter detections. The response can
    include:

    1. The structured JSON detection results (`data`).
    2. The original image dimensions (`origin_image_size`).
    3. The full detection image encoded as Base64 (`base64_result_image`), if requested.
    4. Cropped detected objects encoded as Base64, if requested.

    Args:
        file (UploadFile): The uploaded image file to process.
        conf_threshold (float): Confidence threshold (0-1) to filter detections.
        return_base64_result (bool): If True, includes the full detection image encoded as Base64.
        return_base64_cropped_objects (bool): If True, includes cropped detected objects encoded as Base64.

    Returns:
        JSONResponse: JSON object containing:
            - data: Detection results with objects and bounding boxes.
            - origin_image_size: Original image dimensions.
            - base64_result_image (optional): Full detection image in Base64.
            - base64_cropped_objects (optional): Cropped object images in Base64.

    Raises:
        HTTPException: If the uploaded file is invalid or cannot be processed.
    """
    try:
        image = Image.open(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    async with model_locks["coco"]:
        res = await asyncio.to_thread(
            my_models["coco"], image, conf=conf_threshold, verbose=False
        )
    
    response = process_yolo_result(
        result=res[0],
        return_base64_cropped_objects=return_base64_cropped_objects
    )
    response = {
        "data": response,
        "origin_image_size": {
            "x": res[0].orig_shape[1],
            "y": res[0].orig_shape[0],
        }
    }
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="JPEG")
        buffer.seek(0)
        base64_image = await asyncio.to_thread(
            lambda: base64.b64encode(buffer.getvalue()).decode('utf-8')
        )
        response["base64_result_image"] = base64_image

    return JSONResponse(response)


@app.post("/gun_process_n_return_result", tags=["Firearm Classification"])
async def gun_process_n_return_result(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
    return_base64_result: bool = Form(),
    return_base64_cropped_objects: bool = Form(...),
):
    """
    Perform firearm detection on an uploaded image and optionally return results as Base64.

    This endpoint processes an uploaded image using the YOLO firearm classification model.
    It applies the specified confidence threshold to filter detections. The response can
    include:

    1. The structured JSON detection results (`data`).
    2. The original image dimensions (`origin_image_size`).
    3. The full detection image encoded as Base64 (`base64_result_image`), if requested.
    4. Cropped detected firearm objects encoded as Base64, if requested.

    Args:
        file (UploadFile): The uploaded image file to process.
        conf_threshold (float): Confidence threshold (0-1) to filter detected firearms.
        return_base64_result (bool): If True, includes the full detection image encoded as Base64.
        return_base64_cropped_objects (bool): If True, includes cropped detected firearm objects encoded as Base64.

    Returns:
        JSONResponse: JSON object containing:
            - `data`: Detection results with firearms and bounding boxes.
            - `origin_image_size`: Original image dimensions.
            - `base64_result_image` (optional): Full detection image in Base64.
            - `base64_cropped_objects` (optional): Cropped object images in Base64.

    Raises:
        HTTPException: If the uploaded file is invalid or cannot be processed.
    """
    try:
        image = Image.open(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    async with model_locks["gun"]:
        res = await asyncio.to_thread(
            my_models["gun"], image, conf=conf_threshold, verbose=False
        )
    
    response = process_yolo_result(
        result=res[0],
        return_base64_cropped_objects=return_base64_cropped_objects
    )
    response = {
        "data": response,
        "origin_image_size": {
            "x": res[0].orig_shape[1],
            "y": res[0].orig_shape[0],
        }
    }
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="JPEG")
        buffer.seek(0)
        base64_image = await asyncio.to_thread(
            lambda: base64.b64encode(buffer.getvalue()).decode('utf-8')
        )
        response["base64_result_image"] = base64_image

    return JSONResponse(response)


@app.post("/obj_process_base64", tags=["Object Detection"])
async def obj_process_base64(request: JSONRequest):
    """
    Perform object detection on a base64-encoded image and return structured results.

    This endpoint accepts a JSON payload containing a base64-encoded image and a 
    confidence threshold. The image is decoded and processed using the YOLO COCO 
    object detection model. The detection results are returned as structured JSON.

    Args:
        request (JSONRequest): A JSON payload containing:
            - base64_string (str): The base64-encoded image data.
            - conf_threshold (float): Confidence threshold (0-1) for object detection.

    Returns:
        JSONResponse: Structured JSON containing:
            - Detected objects with bounding boxes and labels.
            - Counts or other processed metadata from `process_yolo_result`.

    Raises:
        HTTPException: If the base64 string is invalid, the image cannot be processed, 
                    or required fields are missing.
    """
    image_data = await asyncio.to_thread(
        base64.b64decode, request.base64_string
    )
    try:
        image = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(image_data))
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    conf_threshold = request.conf_threshold

    async with model_locks["coco"]:
        res = await asyncio.to_thread(
            my_models["coco"], image, conf=conf_threshold, verbose=False
        )

    response = process_yolo_result(res[0])
    return JSONResponse(response)


@app.post("/gun_process_base64", tags=["Firearm Classification"])
async def gun_process_base64(request: JSONRequest):
    """
    Perform firearm detection on a base64-encoded image and return structured results.

    This endpoint accepts a JSON payload containing a base64-encoded image and a 
    confidence threshold. The image is decoded and processed using the YOLO firearm 
    classification model. The detection results are returned as structured JSON.

    Args:
        request (JSONRequest): A JSON payload containing:
            - base64_string (str): The base64-encoded image data to be analyzed.
            - conf_threshold (float): Confidence threshold (0-1) for firearm detection.

    Returns:
        JSONResponse: Structured JSON containing:
            - Detected firearm objects with bounding boxes and labels.
            - Counts or other processed metadata from `process_yolo_result`.

    Raises:
        HTTPException: If the base64 string is invalid, the image cannot be processed,
                    or model inference fails.
    """
    image_data = await asyncio.to_thread(
        base64.b64decode, request.base64_string
    )
    try:
        image = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(image_data))
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    conf_threshold = request.conf_threshold

    async with model_locks["gun"]:
        res = await asyncio.to_thread(
            my_models["gun"], image, conf=conf_threshold, verbose=False
        )

    response = process_yolo_result(res[0])
    return JSONResponse(response)


@app.post("/obj_process_plot_base64", tags=["Object Detection"])
async def obj_process_plot_base64(request: JSONRequest):
    """
    Perform object detection on a base64-encoded image and return a plotted JPEG image.

    This endpoint accepts a JSON payload containing a base64-encoded image and a 
    confidence threshold. The image is decoded, processed using the YOLO COCO 
    object detection model, and the resulting detections are visualized with 
    bounding boxes and labels. The annotated image is returned as a JPEG 
    streaming response.

    Args:
        request (JSONRequest): A JSON payload containing:
            - base64_string (str): The base64-encoded image data.
            - conf_threshold (float): Confidence threshold (0-1) for object detection.

    Returns:
        StreamingResponse: A streaming JPEG image containing the visualized detection results, 
                        including bounding boxes and class annotations.

    Raises:
        HTTPException: If the base64 string is invalid, the image cannot be processed, 
                    or model inference fails.
    """
    image_data = await asyncio.to_thread(
        base64.b64decode, request.base64_string
    )
    try:
        image = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(image_data))
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    conf_threshold = request.conf_threshold

    async with model_locks["coco"]:
        res = await asyncio.to_thread(
            my_models["coco"], image, conf=conf_threshold, verbose=False
        )
    
    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")


@app.post("/gun_process_plot_base64", tags=["Firearm Classification"])
async def gun_process_plot_base64(request: JSONRequest):
    """
    Perform firearm detection on a base64-encoded image and return the visualized result as a JPEG image.

    This endpoint accepts a JSON payload containing a base64-encoded image and a confidence threshold. 
    The image is decoded, processed using the YOLO firearm detection model, and annotated with bounding 
    boxes and labels. The annotated image is then returned as a JPEG streaming response.

    Args:
        request (JSONRequest): A JSON payload containing:
            - base64_string (str): The base64-encoded image data to analyze.
            - conf_threshold (float): Confidence threshold (0-1) for firearm detection.

    Returns:
        StreamingResponse: A streaming JPEG image containing the visualized firearm detection results, 
                        including bounding boxes and classification labels.

    Raises:
        HTTPException: If the base64 string is invalid, the image cannot be processed, 
                    or model inference fails.
    """
    image_data = await asyncio.to_thread(
        base64.b64decode, request.base64_string
    )
    try:
        image = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(image_data))
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    conf_threshold = request.conf_threshold

    async with model_locks["gun"]:
        res = await asyncio.to_thread(
            my_models["gun"], image, conf=conf_threshold, verbose=False
        )

    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="JPEG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")


@app.post("/obj_process_n_return_result_base64", tags=["Object Detection"])
async def obj_process_n_return_result_base64(request: JSONRequest2):
    """
    Perform object detection on a Base64-encoded image and optionally return the annotated result image.

    This endpoint decodes a Base64-encoded image, runs object detection using the YOLO COCO model, 
    and returns structured detection data. Optionally, it can include:
    - Base64-encoded crops of detected objects.
    - A Base64-encoded visualization of the annotated image (with bounding boxes and labels).

    Args:
        request (JSONRequest2): A JSON payload containing:
            - base64_string (str): Base64-encoded image data.
            - conf_threshold (float): Confidence threshold (0-1) for filtering detections.
            - return_base64_result (bool): If True, include the annotated image as a Base64-encoded string.
            - return_base64_cropped_objects (bool): If True, include cropped detected objects as Base64 strings.

    Returns:
        JSONResponse: A JSON object containing:
            - data: Structured detection results with object names, counts, and bounding boxes.
            - origin_image_size: Original image dimensions (`x`, `y`).
            - base64_result_image (optional): Base64-encoded annotated image, if `return_base64_result` is True.

    Raises:
        HTTPException:
            - 400: If the Base64 string is invalid or the image cannot be decoded.
            - 500: If model inference or image encoding fails.
    """
    image_data = await asyncio.to_thread(
        base64.b64decode, request.base64_string
    )
    try:
        image = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(image_data))
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    conf_threshold = request.conf_threshold
    return_base64_cropped_objects = request.return_base64_cropped_objects
    return_base64_result = request.return_base64_result

    async with model_locks["coco"]:
        res = await asyncio.to_thread(
            my_models["coco"], image, conf=conf_threshold, verbose=False
        )
    
    response = process_yolo_result(
        result=res[0],
        return_base64_cropped_objects=return_base64_cropped_objects
    )
    response = {
        "data": response,
        "origin_image_size": {
            "x": res[0].orig_shape[1],
            "y": res[0].orig_shape[0],
        }
    }
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="JPEG")
        buffer.seek(0)
        base64_image = await asyncio.to_thread(
            lambda: base64.b64encode(buffer.getvalue()).decode('utf-8')
        )
        response["base64_result_image"] = base64_image

    return JSONResponse(response)


@app.post("/gun_process_n_return_result_base64", tags=["Firearm Classification"])
async def gun_process_n_return_result_base64(request: JSONRequest2):
    """
    Perform firearm classification on an image provided as a Base64 string using a YOLO model.

    Args:
        request (JSONRequest2): A JSON object containing:
            - `base64_string` (str): The Base64-encoded string representing the input image.
            - `conf_threshold` (float): The confidence threshold for filtering detections (between 0 and 1).
            - `return_base64_result` (bool): Whether to include the processed result image as a Base64-encoded string.
            - `return_base64_cropped_objects` (bool): Whether to return cropped detected objects as Base64 strings.

    Returns:
        JSONResponse: A JSON object containing:
            - `data`: The structured detection results, including detected objects and their bounding boxes.
            - `origin_image_size`: The original image dimensions (`x` and `y`).
            - `base64_result_image` (optional): The processed image encoded in Base64, if `return_base64_result` is True.

    Raises:
        HTTPException: If there are issues with the input Base64 string or processing the image.
    """
    image_data = await asyncio.to_thread(
        base64.b64decode, request.base64_string
    )
    try:
        image = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(image_data))
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    conf_threshold = request.conf_threshold
    return_base64_cropped_objects = request.return_base64_cropped_objects
    return_base64_result = request.return_base64_result

    async with model_locks["gun"]:
        res = await asyncio.to_thread(
            my_models["gun"], image, conf=conf_threshold, verbose=False
        )
    
    response = process_yolo_result(
        result=res[0],
        return_base64_cropped_objects=return_base64_cropped_objects
    )
    response = {
        "data": response,
        "origin_image_size": {
            "x": res[0].orig_shape[1],
            "y": res[0].orig_shape[0],
        }
    }
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="JPEG")
        buffer.seek(0)
        base64_image = await asyncio.to_thread(
            lambda: base64.b64encode(buffer.getvalue()).decode('utf-8')
        )
        response["base64_result_image"] = base64_image

    return JSONResponse(response)


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
            "Firearm Classification Models": gun_type,
        }
    )


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
    async with model_locks["coco"]:
        del my_models["coco"]
        gc.collect()
        empty_cache()
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
    async with model_locks["gun"]:
        del my_models["gun"]
        gc.collect()
        empty_cache()
        my_models["gun"] = YOLO(gun_pth)
    return JSONResponse(
        {"message": f"Model {model_type.name} is selected"}
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
    async with model_locks["coco"]:
        del my_models["coco"]
        gc.collect()
        empty_cache()
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
    async with model_locks["gun"]:
        del my_models["gun"]
        gc.collect()
        empty_cache()
        my_models["gun"] = YOLO(gun_pth)
    return JSONResponse(
        {"message": f"Model from path {request.model_type} is selected"}
    )


uvicorn.run(app, host="0.0.0.0", port=8080)