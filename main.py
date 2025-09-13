import os
import sys
import json
import base64
import mimetypes
import asyncio
import tempfile
from io import BytesIO
from typing import List, Dict, Any

# --- Core Dependencies ---
import pdfplumber
from PIL import Image
from groq import AsyncGroq # Use the Asynchronous client for FastAPI

# --- FastAPI Dependencies ---
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access variables
api_key = os.getenv("GROQ_API_KEY")
# ---------- CONFIG & API CLIENT ----------
# For reliability, using standard Groq models. You can change these via environment variables.
TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "qwen/qwen3-32b")  # text-only on Groq
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")  # image+text

# Initialize the Asynchronous Groq client.
try:
    client = AsyncGroq()
except Exception as e:
    print(f"FATAL: Failed to initialize Groq client. Is GROQ_API_KEY set? Error: {e}")
    sys.exit(1)

# The target schema for our intelligent form filler.
INSURANCE_FORM_FIELDS = [
    "Full Name", "Address", "Phone Number", "Email", "Policy Number",
    "Type of Claim", "Date of Incident", "Location of Incident",
    "Description of Incident", "Details of Loss/Damage/Injury",
    "Were there any injuries? If yes, provide details", "Supporting Documents"
]

# ---------- FASTAPI APP INITIALIZATION ----------
app = FastAPI(
    title="📄 Intelligent Document Processing API",
    description="An API with two main features: `/extract/` for raw data extraction and `/fill-form/` for intelligent form completion.",
    version="2.0.0",
)

# Add CORS middleware to allow frontend requests.
# For production, you should restrict this to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SHARED UTILITY FUNCTIONS ----------
IMAGE_EXTS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

def is_image_file(filename: str) -> bool:
    """Checks if a filename has a common image extension."""
    return filename.lower().split(".")[-1] in IMAGE_EXTS

def encode_image_as_data_url(image_path: str) -> str:
    """Encodes an image file as a base64 data URL, resizing if it exceeds 4MB."""
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    with open(image_path, "rb") as f:
        raw = f.read()

    max_bytes = 4 * 1024 * 1024
    if len(raw) <= max_bytes:
        return f"data:{mime};base64," + base64.b64encode(raw).decode("utf-8")
    
    # If the image is too large, attempt to downscale it
    img = Image.open(BytesIO(raw)).convert("RGB")
    quality = 85
    w, h = img.size
    for _ in range(8):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return f"data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")
        quality = max(50, quality - 10)
        w, h = int(w * 0.85), int(h * 0.85)
        img = img.resize((max(1, w), max(1, h)), Image.Resampling.LANCZOS)
    return f"data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts all text content from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text(x_tolerance=2) or ""
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text.strip()

# ---------- ASYNC LLM & PROCESSING FUNCTIONS ----------

async def extract_raw_info_from_doc(file_path: str, filename: str) -> dict:
    """Extracts raw key-value pairs from a document using the appropriate Groq LLM."""
    try:
        if is_image_file(filename):
            model = VISION_MODEL
            content = [
                {"type": "text", "text": "Extract all key-value pairs from this document. Respond with only a valid, flat JSON object."},
                {"type": "image_url", "image_url": {"url": encode_image_as_data_url(file_path)}},
            ]
        elif filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
            if not text:
                return {"warning": f"No text could be extracted from {filename}."}
            model = TEXT_MODEL
            content = f"Extract all key-value pairs from the following text. Respond with only a valid, flat JSON object:\n\n{text}"
        else:
            return {"error": f"Unsupported file type: {filename}"}

        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that extracts information into a single, flat JSON object."},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"error": f"LLM extraction failed for {filename}: {str(e)}"}

async def fill_missing_fields_with_llm(missing_fields: list, source_data: dict) -> dict:
    """Uses an LLM to intelligently map combined extracted source data to missing form fields."""
    if not missing_fields or not source_data:
        return {}

    prompt = f"""
You are an intelligent data mapping assistant. Your task is to fill in missing insurance form fields using data aggregated from one or more documents.

**1. Missing Fields from the Insurance Form:**
```json
{json.dumps(missing_fields, indent=2)}
```

**2. Combined Data Extracted from All Provided Documents:**
```json
{json.dumps(source_data, indent=2)}
```

**Instructions:**
- Analyze the source data and map it to the semantically correct missing fields.
- Return a JSON object containing ONLY the fields from the missing list that you were able to confidently fill.
- If the source data is irrelevant or cannot fill any missing fields, return an empty JSON object {{}}.
"""
    try:
        resp = await client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a data mapping expert that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM mapping failed: {e}")
        return {}

# ---------- API ENDPOINTS ----------

@app.get("/", tags=["Root"])
async def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Document Processing API. Visit /docs for details."}

@app.get("/form-schema", tags=["Form Filling"])
async def get_form_schema():
    """Returns the JSON schema (list of fields) for the insurance form."""
    return {"fields": INSURANCE_FORM_FIELDS}

@app.post("/extract/", tags=["Extraction"])
async def extract_information_from_files(files: List[UploadFile] = File(...)):
    """
    Uploads one or more documents and performs raw key-value extraction on each.
    Returns a dictionary with filenames as keys and extracted data as values.
    """
    async def process_single_file(file: UploadFile):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        try:
            return await extract_raw_info_from_doc(temp_path, file.filename)
        finally:
            os.remove(temp_path)

    tasks = [process_single_file(file) for file in files]
    results_list = await asyncio.gather(*tasks)
    return {files[i].filename: results_list[i] for i in range(len(files))}


@app.post("/fill-form/", tags=["Form Filling"])
async def intelligently_fill_form(
    files: List[UploadFile] = File(...),
    current_form_state: str = Form(...)
):
    """
    Uploads one or more documents and the current state of a form. It extracts data
    from all documents, combines it, and uses an LLM to fill in the missing fields.
    Returns the complete, updated form state. This is the one-shot filling process.
    """
    try:
        form_dict = json.loads(current_form_state)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for current_form_state.")

    # --- Step 1: Extract data from all uploaded files concurrently ---
    async def process_file_for_filling(file: UploadFile):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        try:
            return file.filename, await extract_raw_info_from_doc(temp_path, file.filename)
        finally:
            os.remove(temp_path)

    print(f"📄 Processing {len(files)} document(s)...")
    extraction_tasks = [process_file_for_filling(file) for file in files]
    extraction_results = await asyncio.gather(*extraction_tasks)
    
    # --- Step 2: Combine all extracted data into a single dictionary ---
    combined_extracted_data = {}
    source_documents = {}
    for filename, data in extraction_results:
        source_documents[filename] = data
        if data and "error" not in data and "warning" not in data:
            combined_extracted_data.update(data)

    if not combined_extracted_data:
        return JSONResponse(
            status_code=422,
            content={
                "message": "Could not extract any usable data from the provided document(s).",
                "source_documents": source_documents,
                "updated_form": form_dict
            }
        )
    
    # --- Step 3: Identify missing fields and call the intelligent filler ---
    missing_fields = [k for k, v in form_dict.items() if not v]
    if not missing_fields:
        return {
            "message": "No fields were missing. No update needed.",
            "source_documents": source_documents,
            "updated_form": form_dict
        }
        
    print("🧠 Asking LLM to fill missing fields with combined data...")
    newly_filled_data = await fill_missing_fields_with_llm(missing_fields, combined_extracted_data)
    
    # --- Step 4: Update the form state and prepare the response ---
    updated_form_dict = form_dict.copy()
    if newly_filled_data:
        for key, value in newly_filled_data.items():
            if key in missing_fields: # Only update fields that were originally empty
                updated_form_dict[key] = value
        message = f"Successfully attempted to fill {len(newly_filled_data)} field(s)."
    else:
        message = "The documents did not contain relevant information for the missing fields."
    
    return {
        "message": message,
        "newly_filled_data": newly_filled_data,
        "updated_form": updated_form_dict,
        "source_documents": source_documents,
    }
