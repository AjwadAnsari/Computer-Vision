from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from io import BytesIO
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (3,3), 0)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred_image, 100, 200)
    
    # Encode processed image to send as response
    _, img_encoded = cv2.imencode(".png", edges)
    img_bytes = BytesIO(img_encoded.tobytes())
    return StreamingResponse(img_bytes, media_type="image/png")
  
