cd from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import json
from collections import defaultdict
from paddleocr import PPStructureV3

app = FastAPI()

# Initialize pipeline
pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    device="cpu"
)

@app.post("/extract-json/")
async def extract_json(file: UploadFile = File(...)):
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    # Save uploaded image to temp path
    temp_image_path = os.path.join(temp_dir, file.filename)
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run OCR pipeline
        output = pipeline.predict(input=temp_image_path)

        # Save outputs to temp folder
        for res in output:
            res.save_to_json(save_path=temp_dir)
            res.save_to_markdown(save_path=temp_dir)
            res.save_to_img(save_path=temp_dir)

        # Find the saved JSON filename
        base_filename = os.path.splitext(os.path.basename(temp_image_path))[0]
        result_json_path = os.path.join(temp_dir, f"{base_filename}_res.json")

        # Read the JSON and group boxes
        with open(result_json_path, "r") as f:
            data = json.load(f)

        grouped_boxes = defaultdict(list)
        for item in data.get("layout_det_res", {}).get("boxes", []):
            category = item.get("label")
            bbox = item.get("coordinate")
            grouped_boxes[category].append(bbox)

        return JSONResponse(content=grouped_boxes)

    finally:
        # Clean up all files in the temp folder
        # Cleanup safely — remove files and directories inside temp_dir
        for f in os.listdir(temp_dir):
            path = os.path.join(temp_dir, f)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Error deleting {path}: {e}")
