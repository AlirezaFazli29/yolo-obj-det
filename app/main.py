from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from model_handler import YoloType
from ultralytics import YOLO
from utils import process_yolo_result
from PIL import Image
import base64
import uvicorn


app = FastAPI(title="object detection")

yolo_coco = YOLO(YoloType.Pretrained.yolo11n.value)
# yolo_gun = YOLO(YoloType.Custom.Firearm_best.value)


@app.get("/")
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


@app.post("/file-to-base64")
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


@app.post("/obj_process")
async def obj_process(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    
    image = Image.open(file.file)
    res = yolo_coco(image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])

    return response


uvicorn.run(app, host="0.0.0.0", port=8080)