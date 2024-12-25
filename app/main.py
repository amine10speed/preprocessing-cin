from fastapi import FastAPI, File, UploadFile
import shutil
from app.preprocessing import preprocess_image
import cv2
import os

app = FastAPI()

@app.post("/preprocess/")
async def preprocess_endpoint(file: UploadFile = File(...), debug: bool = False):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(file_location, debug=debug)

        # Save the preprocessed image
        preprocessed_path = f"preprocessed_{file.filename}"
        cv2.imwrite(preprocessed_path, preprocessed_image)

        os.remove(file_location)  # Clean up the temporary file
        return {"message": "Image preprocessed successfully", "preprocessed_image": preprocessed_path}
    except Exception as e:
        os.remove(file_location)
        return {"message": "Error during preprocessing", "error": str(e)}
