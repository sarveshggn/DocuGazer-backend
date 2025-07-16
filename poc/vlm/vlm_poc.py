import asyncio
import base64
import json
import uuid
from io import BytesIO
from threading import Lock
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pdf2image import convert_from_bytes
from pydantic import BaseModel
from PIL import Image
import os
import cv2
import numpy as np

app = FastAPI()

# In-memory session storage
sessions: Dict[str, List[str]] = {}
sessions_lock = Lock()

OLLAMA_URL = "http://localhost:11434/api/generate"


# Declare Pydantic model for better input handling and verification
class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = "qwen3:4b"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    session_id: Optional[str] = None
    stream: Optional[bool] = False  # TODO: Add functionality for streaming


def process_image(image_bytes: bytes, prompt: str, model: str) -> str:
    """
    Process an image with a prompt and return the response from the OLLAMA service.

    :param image_bytes: The image data in bytes\n
    :param prompt: The user's prompt\n
    :param model: The vision model to use\n
    :return: The response text from the OLLAMA service\n
    """
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()
    return result.get("response")

def extract_layout(image_bytes: bytes) -> dict:
    """
    Extract layout information from an image using the PPStructure API.

    :param image_bytes: The image data in bytes\n
    :return: Dictionary containing layout analysis results\n
    """
    files = {"file": ("image.png", image_bytes, "image/png")}
    response = requests.post("http://localhost:7001/extract-json/", files=files)
    
    if response.status_code == 200:
        layout_data = response.json()
        return layout_data
    else:
        raise Exception(f"Failed to extract layout data: {response.status_code}")


def crop_image_regions(image_bytes: bytes, layout_data: dict, output_dir: str = "cropped_regions") -> Dict[str, List[str]]:
    """
    Crop image regions based on layout analysis data and save them to files.
    
    :param image_bytes: The original image data in bytes
    :param layout_data: JSON response from Paddle containing layout analysis
    :param output_dir: Directory to save cropped images
    :return: Dictionary mapping region types to lists of saved file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert bytes to numpy array for OpenCV (for cropping only, no display)
    nparr = np.frombuffer(image_bytes, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if cv_image is None:
        return {}
    
    # Also open with PIL for cropping
    image = Image.open(BytesIO(image_bytes))
    
    # Dictionary to store file paths for each region type
    cropped_files = {}
    
    # Get the layout analysis data - handle both possible structures
    if "layout_analysis" in layout_data:
        layout_analysis = layout_data["layout_analysis"]
    else:
        # If layout_data itself contains the region data directly
        layout_analysis = layout_data
    
    # Process each region type
    for region_type, coordinates_list in layout_analysis.items():
        if not isinstance(coordinates_list, list):
            continue
            
        cropped_files[region_type] = []
        
        for i, coordinates in enumerate(coordinates_list):
            if len(coordinates) != 4:
                continue
                
            # Extract coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = coordinates
            
            # Convert to integers and ensure they're within image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.width, int(x2))
            y2 = min(image.height, int(y2))
            
            # Skip if the region is invalid
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Crop the image using PIL
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Generate filename
            filename = f"{region_type}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save the cropped image
            cropped_image.save(filepath, "PNG")
            cropped_files[region_type].append(filepath)
    
    return cropped_files


@app.get("/")
async def root():
    return {"message": "Ollama FastAPI API with session management is running."}





@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Endpoint to process a text input prompt

    :param req: User request containing prompt and optional variables like model, temperature, top_p, session_id, streaming_output\n
    :return: Text response generated by the LLM for the user query
    """
    session_id = req.session_id or str(uuid.uuid4())

    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(f"User: {req.prompt}")

    payload = {
        "model": req.model,
        "prompt": "\n".join(sessions[session_id]),
        "options": {
            "temperature": req.temperature,
            "top_p": req.top_p
        },
        "stream": req.stream
    }

    if req.stream:
        def event_stream():
            response = requests.post(OLLAMA_URL, json=payload, stream=True)
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8") + "\n"

        return StreamingResponse(event_stream(), media_type="text/plain")

    else:
        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json()

        with sessions_lock:
            sessions[session_id].append(f"AI: {result.get('response')}")

        return JSONResponse(content={"response": result.get("response"), "session_id": session_id})


@app.get("/session/{session_id}")
async def get_session_history(session_id: str):
    """
    Endpoint to view session history

    :param session_id: The session id for which the user wants to see the history\n
    :return: JSON object containing the user's session history
    """
    with sessions_lock:
        history = sessions.get(session_id)
        if history is None:
            return JSONResponse(content={"error": "Session not found"}, status_code=404)
    return {"session_id": session_id, "history": history}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Endpoint to delete session

    :param session_id: The session id that the user wants to delete\n
    :return: Acknowledgement if the session was deleted or not
    """
    with sessions_lock:
        if session_id in sessions:
            del sessions[session_id]
            return {"message": f"Session {session_id} deleted."}
        else:
            return JSONResponse(content={"error": "Session not found"}, status_code=404)


@app.post("/generate_image_response")
async def generate_image_response(
        image: UploadFile,
        prompt: str = Form(...),
        model: Optional[str] = Form("qwen2.5vl:7b-fp16"),
        crop_regions: Optional[bool] = Form(False)
):
    """
    Endpoint to process a single image with a prompt.

    :param image: Input image the user wants to query on\n
    :param prompt: Prompt from the user\n
    :param model: The vision model the user wants to use\n
    :param crop_regions: Whether to crop and save detected regions\n
    :return: Text response to the user generated from the VLM\n
    """
    # TODO: Add session history
    image_bytes = await image.read()
    
    try:
        layout_data = extract_layout(image_bytes)
        
        # Crop regions if requested
        cropped_files = {}
        if crop_regions:
            output_dir = "cropped_regions_single_image"
            cropped_files = crop_image_regions(image_bytes, layout_data, output_dir)
        
        result = {"layout_analysis": layout_data}
        
        if crop_regions:
            result["cropped_regions"] = cropped_files
            
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/generate_pdf_response")
async def generate_pdf_response(
        pdf: UploadFile,
        prompt: str = Form(...),
        model: Optional[str] = Form("qwen2.5vl:7b-fp16"),
        crop_regions: Optional[bool] = Form(False)
):
    """
    Endpoint to process a PDF with a prompt.

    :param pdf: The uploaded PDF file\n
    :param prompt: Prompt from the user\n
    :param model: The vision model the user wants to use\n
    :param crop_regions: Whether to crop and save detected regions\n
    :return: JSON response with results for each page\n
    """
    # TODO: Add session history
    pdf_bytes = await pdf.read()
    images = convert_from_bytes(pdf_bytes)

    results = []
    for page_num, image in enumerate(images, start=1):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        try:
            layout_data = extract_layout(image_bytes)
            
            # Crop regions if requested
            cropped_files = {}
            if crop_regions:
                output_dir = f"cropped_regions_page_{page_num}"
                cropped_files = crop_image_regions(image_bytes, layout_data, output_dir)
            
            result = {
                "page": page_num, 
                "layout_analysis": layout_data
            }
            
            if crop_regions:
                result["cropped_regions"] = cropped_files
                
            results.append(result)
        except Exception as e:
            results.append({"page": page_num, "error": str(e)})

    return JSONResponse(content={"results": results})


# @app.post("/generate_pdf_streaming_response")
# async def generate_pdf_streaming_response(
#         pdf: UploadFile,
#         prompt: str = Form(...),
#         model: Optional[str] = Form("qwen2.5vl:7b-fp16")
# ):
#     """
#     Endpoint to process each page of a PDF with a prompt, streaming results as they are generated.
#
#     :param pdf: The uploaded PDF file\n
#     :param prompt: Prompt from the user\n
#     :param model: The vision model the user wants to use\n
#     :return: Streaming response with results for each page\n
#     """
#
#     async def stream_results():
#         pdf_bytes = await pdf.read()
#         images = convert_from_bytes(pdf_bytes)
#
#         for page_num, image in enumerate(images, start=1):
#             buffer = BytesIO()
#             image.save(buffer, format="PNG")
#             image_bytes = buffer.getvalue()
#             response_text = process_image(image_bytes, prompt, model)
#             yield json.dumps({"page": page_num, "response": response_text}) + "\n"
#             await asyncio.sleep(0.1)
#
#     return StreamingResponse(
#         stream_results(),
#         media_type="application/x-ndjson"
#     )
