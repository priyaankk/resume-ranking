from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import openai
from dotenv import load_dotenv
import os
import requests
import time
import docx
import io
import asyncio
from io import BytesIO
import json
from colorama import init, Fore, Back, Style

# Load environment variables
load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")

app = FastAPI()

# Configure Azure OpenAI API
client = openai.AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version="2023-07-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

async def extract_text_from_pdf(pdf_content: bytes) -> str:
    # Extracts text from PDF using Azure Document Intelligence
    print(Style.BRIGHT + Fore.YELLOW + "Extracting text from PDF using Azure Document Intelligence" + Style.RESET_ALL)
    headers = {
        "Ocp-Apim-Subscription-Key": DOCUMENT_INTELLIGENCE_KEY,
        "Content-Type": "application/pdf"
    }

    # Submit PDF for processing
    response = requests.post(
        f"{DOCUMENT_INTELLIGENCE_ENDPOINT}/formrecognizer/documentModels/prebuilt-read:analyze?api-version=2023-07-31",
        headers=headers,
        data=pdf_content
    )

    if response.status_code != 202:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")

    # Get operation URL and poll for results
    operation_url = response.headers["Operation-Location"]
    
    while True:
        result_response = requests.get(
            operation_url, 
            headers={"Ocp-Apim-Subscription-Key": DOCUMENT_INTELLIGENCE_KEY}
        )
        result_json = result_response.json()

        if result_json.get("status") == "succeeded":
            return result_json["analyzeResult"]["content"]
        elif result_json.get("status") == "failed":
            raise Exception(f"Processing failed: {result_json}")
        
        await asyncio.sleep(5)

def extract_text_from_docx(file: bytes) -> str:
    # Extracts text from DOCX using docx module
    print(Fore.YELLOW + "Extracting text from DOCX using docx module" + Style.RESET_ALL)
    docx_file = io.BytesIO(file)
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def generate_criteria(job_description: str) -> dict:
    # Generates criteria using Azure OpenAI
    print(Fore.CYAN + "Generating criteria using Azure OpenAI" + Style.RESET_ALL)
    try:
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant designed to extract **clear, quantifiable, and measurable** job criteria from a given job description. "
                        "Your output must be **objective, specific, and numerically evaluable**, avoiding vague or subjective terms. "
                        "Each extracted criterion should explicitly state measurable attributes, such as required years of experience, certifications, degrees, and specific technical skills with proficiency levels. "
                        "Avoid using ambiguous phrases like 'strong knowledge', 'expertise', 'familiarity', or 'ability to'. "
                        "Do **not** omit any criteria explicitly mentioned as necessary or binding, such as a minimum required experience. "
                        "Instead, ensure all essential qualifications are captured accurately, e.g., '3+ years of experience in Java', 'AWS Certified Solutions Architect', or 'Master’s degree in Data Science'. "
                        "Your response should be structured as a **valid JSON object** containing a key \"criteria\". "
                        "Do **not** mention the word 'JSON' in the response. "
                        "Ensure that the output strictly follows this format: "
                        "{'criteria': ["
                        "'5+ years of experience in Machine Learning', "
                        "'Master’s degree in Computer Science, AI, or related field', "
                        "'AWS Certified Machine Learning Specialty (preferred)', "
                        "'Proficiency in Python (Advanced)', "
                        "'2+ years of experience with TensorFlow and PyTorch', "
                        "]}"
                    )
},
                {"role": "user", "content": job_description}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )
                
        try:
            parsed_response = json.loads(response.choices[0].message.content)
            if not isinstance(parsed_response, dict) or 'criteria' not in parsed_response:
                return {"error": "Invalid response format", "criteria": []}
            return parsed_response
        except json.JSONDecodeError as je:
            return {"error": f"JSON parsing error: {str(je)}", "criteria": []}
            
    except Exception as e:
        return {"error": f"API error: {str(e)}", "criteria": []}

@app.post(
    "/extract-criteria/",
    tags=["Upload"],
    summary="Upload PDF or DOCX file",
    description="Upload a PDF or DOCX file to extract job criteria",
    response_description="Returns extracted job criteria as a JSON object"
)
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    try:
        if file.filename.endswith('.pdf'):
            text = await extract_text_from_pdf(content)
        elif file.filename.endswith('.docx'):
            text = extract_text_from_docx(content)
        else:
            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from {file.filename}: {str(e)}")
    
    criteria = generate_criteria(text)
    
    return criteria

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
    