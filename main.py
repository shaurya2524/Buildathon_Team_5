import os
import sys
import json
import base64
import mimetypes
import asyncio
import tempfile
import re
from io import BytesIO
from typing import List, Dict, Any

# --- Core Dependencies ---
import pdfplumber
from PIL import Image
from groq import AsyncGroq, Groq
from twilio.rest import Client

# --- FastAPI Dependencies ---
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ---------- CONFIG & API CLIENTS ----------
# For reliability, using standard Groq models. You can change these via environment variables.
TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "qwen/qwen3-32b")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Initialize the Asynchronous Groq client for document processing.
try:
    async_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"FATAL: Failed to initialize AsyncGroq client. Is GROQ_API_KEY set? Error: {e}")
    sys.exit(1)

# Initialize Synchronous Groq client for WhatsApp message generation
try:
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
except Exception as e:
    print(f"FATAL: Failed to initialize Groq client. Is GROQ_API_KEY set? Error: {e}")
    sys.exit(1)

# Initialize Twilio client
try:
    twilio_client = Client(
        os.environ["TWILIO_ACCOUNT_SID"],
        os.environ["TWILIO_AUTH_TOKEN"]
    )
except Exception as e:
    print(f"FATAL: Failed to initialize Twilio client. Are TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN set? Error: {e}")
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
    description="An API with features for data extraction, form completion, claim letter generation, and notifications.",
    version="2.1.0",
)

# Add CORS middleware to allow frontend requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Models for Request Bodies ----------
class PolicyDetails(BaseModel):
    policyName: str = Field(..., example="Health Insurance")
    policyNumber: str = Field(..., example="PC_001")
    insuranceCompany: str = Field(..., example="Acme Insurance")

class RenewalMessagePayload(BaseModel):
    customerName: str = Field(..., example="John Doe")
    policy: PolicyDetails
    renewalDate: str = Field(..., example="25th September 2025")
    contact: str = Field(..., example="+919876543210", description="Recipient's full number with country code")

class ClaimLetterPayload(BaseModel):
    extracted_data: Dict[str, Dict[str, Any]] = Field(..., example={
        "document1.pdf": {"Policy Number": "12345", "Customer Name": "John Doe", "Insurance Company Name": "Future Generali"},
        "receipt.jpg": {"Total Amount": "5000", "Date of Incident": "2024-08-15"}
    })


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

# ---------- WHATSAPP NOTIFICATION FUNCTIONS ----------

def generate_message(customer_name, policy_name, renewal_date, company_name):
    """
    Generate a professional SMS renewal reminder message with a payment link.
    """
    prompt = f"""
    Write a professional but friendly SMS renewal reminder message (3–4 sentences) 
    for a customer named {customer_name}.
    They hold an insurance policy of type '{policy_name}' with {company_name}.
    Inform them that their renewal date is approaching on {renewal_date}.
    
    Add a dummy payment link in the message, formatted like: 👉 https://pay.link  
    Use short paragraphs separated by \\n (maximum 2 line breaks).  
    Highlight important details using CAPS (since SMS doesn't support bold/italics).  
    Return only the final message inside double quotes.
    """

    response = groq_client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    full_text = response.choices[0].message.content
    print("Raw LLM Output:\n", full_text)

    # Extract longest quoted block
    matches = re.findall(r'"(.*?)"', full_text, re.DOTALL)
    if matches:
        message_text = max(matches, key=len).strip()
    else:
        message_text = full_text.strip()

    # Ensure Twilio-safe formatting
    message_text = message_text.replace("\\n", "\n")

    print("\n✅ Final SMS-ready Message:\n", message_text)
    return message_text


def send_whatsapp(to_number, message_body):
    """
    Send WhatsApp message using Twilio Sandbox.
    """
    try:
        message = twilio_client.messages.create(
            body=message_body,
            from_="whatsapp:+14155238886",  # Twilio WhatsApp sandbox number
            to=f"whatsapp:{to_number}"      # Recipient's WhatsApp number
        )
        print("📩 WhatsApp Message sent! SID:", message.sid)
        return message.sid
    except Exception as e:
        print(f"❌ Failed to send WhatsApp message: {e}")
        raise HTTPException(status_code=500, detail=f"Twilio API Error: {str(e)}")


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

        resp = await async_client.chat.completions.create(
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
        resp = await async_client.chat.completions.create(
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
        
async def generate_combined_claim_letter(all_info: dict) -> str:
    """
    Generate a single, consolidated claim letter from information extracted from multiple files.
    """
    prompt = f"""
    You are an expert insurance assistant. You have been provided with JSON data extracted from multiple documents related to a single insurance claim. Your task is to synthesize this information into one comprehensive and professional claim letter.

    **Formatting Instructions:**
    1.  **Highlighting:** To emphasize the most critical information (like policy numbers, dates, monetary amounts, etc.), enclose it in double asterisks. For example: `The total cost is **₹14,500**.`
    2.  **Attached Documents:** For the 'Documentation Attached' section, write a natural, descriptive sentence about the types of documents attached, based on their filenames and content. **Do not simply list the filenames.** For example, instead of "Attached are policy_form.pdf, fir.pdf...", write something like "For your review, we have attached the user's policy form, the official First Information Report (FIR), and the vehicle service bills."
    3.  **Content:** Intelligently merge details from all documents. If information is missing after reviewing all sources, state 'Not Provided'. The agent's name and details are fixed and provided in the template.
    4.  **Output:** Generate ONLY the letter content, starting directly with "Subject:". Do not include any introductory text or explanations.

    **Letter Template:**

    Subject: Insurance Claim Submission – Policy #**[Policy Number]**

    Dear [Insurance Company Name / Claims Department],

    I am writing to formally submit an insurance claim under policy number **[Policy Number]**, regarding an incident that occurred on **[Date of Incident]**, resulting in losses to [Insured Party / Customer Name].

    I am Vikram Singh, serving as the authorized insurance agent for [Business / Customer Name].

    **Incident Details**

    Date & Time of Incident: **[Date and Time]**
    Location of Incident: [Location]
    Description of Incident: [Clear, concise description of what happened, synthesized from all documents]
    Impact / Losses Sustained: [Summary of damages or losses, synthesized from all documents]
    Mitigation Actions Taken: [Steps taken to reduce further damage]

    **Documentation Attached**
    [Your generated descriptive sentence about the attached documents goes here.]

    **Claim Amount**
    The total estimated losses and associated costs amount to **₹[Total Amount]**. We kindly request reimbursement as per policy terms to support business recovery and continuity.

    **Next Steps**

    We request a prompt review of this claim and will provide any additional information or clarification required. Please confirm receipt of this submission.
    You may contact me directly at **+91 98765 43210** or **vikram.singh@singhinsuranceservices.com** for any follow-up.

    Thank you for your prompt attention to this matter.

    Sincerely,
    Vikram Singh
    Insurance Agent
    Singh Insurance Services

    ---
    **Extracted Information JSON:**
    {json.dumps(all_info, indent=2)}
    """

    try:
        resp = await async_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw_letter_output = resp.choices[0].message.content

        # Clean potential model "thinking" tags
        if "</think>" in raw_letter_output:
            combined_claim_letter = raw_letter_output.split("</think>", 1)[1].strip()
        else:
            combined_claim_letter = raw_letter_output.strip()

        return combined_claim_letter
    except Exception as e:
        print(f"Error generating claim letter: {e}")
        raise HTTPException(status_code=500, detail=f"LLM claim letter generation failed: {str(e)}")


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

@app.post("/generate-claim-letter/", tags=["Claim Generation"])
async def generate_claim_letter_endpoint(payload: ClaimLetterPayload):
    """
    Generates a consolidated claim letter from extracted document data.
    """
    print("📝 Generating claim letter...")
    
    # Filter out any documents that resulted in an error during extraction
    valid_info = {
        file: info for file, info in payload.extracted_data.items() if "error" not in info
    }

    if not valid_info:
        raise HTTPException(status_code=422, detail="No valid information provided. Cannot generate claim letter.")

    try:
        letter_content = await generate_combined_claim_letter(valid_info)
        return {
            "success": True,
            "message": "Claim letter generated successfully.",
            "claim_letter_content": letter_content
        }
    except HTTPException as e:
        # Re-raise exceptions from the generation function
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during claim letter generation: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.post("/send-renewal-reminder/", tags=["WhatsApp Notifications"])
async def send_renewal_reminder(payload: RenewalMessagePayload = Body(...)):
    """
    Generates a personalized insurance renewal reminder and sends it via WhatsApp.
    """
    print(f"Received renewal request for {payload.customerName} at {payload.contact}")
    try:
        # Step 1: Generate the personalized message
        message_body = await asyncio.to_thread(
            generate_message,
            payload.customerName,
            payload.policy.policyName,
            payload.renewalDate,
            payload.policy.insuranceCompany
        )
        
        # Step 2: Send the message via Twilio WhatsApp API
        message_sid = await asyncio.to_thread(
            send_whatsapp,
            payload.contact,
            message_body
        )

        return {
            "success": True,
            "message": "WhatsApp renewal reminder sent successfully.",
            "recipient": payload.contact,
            "twilio_message_sid": message_sid,
            "sent_message": message_body
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions from send_whatsapp
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
