from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import List
import pandas as pd
import PyPDF2
import docx
import io
import json
from pydantic import BaseModel
import tempfile
import os
import openai
from dotenv import load_dotenv
import requests
import asyncio
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sys
import ssl
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from colorama import init, Fore, Back, Style
import json
from openai import OpenAI
import warnings

warnings.simplefilter("ignore", FutureWarning)

# # Initialize colorama (for Windows compatibility)
# init()


# Load environment variables
load_dotenv(override=True)

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Document Intelligence Configuration
DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")

# Add these to environment variables configuration section
AZURE_TEXT_ANALYTICS_ENDPOINT = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")
AZURE_TEXT_ANALYTICS_KEY = os.getenv("AZURE_TEXT_ANALYTICS_KEY")

# Initialize Azure OpenAI client
client = openai.AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version="2023-07-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)


# Initialize Azure Text Analytics Client
text_analytics_client = TextAnalyticsClient(
    endpoint=AZURE_TEXT_ANALYTICS_ENDPOINT, 
    credential=AzureKeyCredential(AZURE_TEXT_ANALYTICS_KEY)
)

app = FastAPI(
    title="Resume Scoring API",
    version="1.0.0",
)


class ScoringResponse(BaseModel):
    candidate_name: str
    scores: dict
    total_score: float

async def extract_text_from_pdf(pdf_content: bytes) -> str:
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
    print(Fore.YELLOW + "Extracting text from DOCX using docx module" + Style.RESET_ALL)
    docx_file = io.BytesIO(file)
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def preprocess_text(text: str) -> str:
    # print("text in preprocess_text before preprocessing: ", text[:100])
    # Convert to lowercase
    text = text.lower()
    # print("text in preprocess_text after converting to lowercase: ", text[:100])
    # Tokenization
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]  # Keep only alphanumeric tokens
    # print("tokens after tokenization: ", tokens[:30])
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    # print("tokens after removing stopwords and non-alphabetic tokens: ", tokens[:30])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # print("tokens after lemmatization: ", tokens[:30])
    # Stemming                                                      # Stemming removed because stemming after lemmatization is not needed and is harmful
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(token) for token in tokens]
    # print("return after preprocessing: ", ' '.join(tokens))
    return ' '.join(tokens)


def calculate_similarity_scores(resume_text: str, criteria: List[str]) -> dict:
    # Initialize SentenceTransformer model (use a pre-trained model for embedding)
    model = SentenceTransformer('all-MiniLM-L6-v2')  

    # Preprocess the resume and criteria
    processed_resume = preprocess_text(resume_text)
    criteria_list = criteria["criteria"]  # Extract the actual list of criteria
    processed_criteria = [preprocess_text(criterion) for criterion in criteria_list]
    
    # Encode the resume and criteria to sentence embeddings
    resume_embedding = model.encode([processed_resume])[0]  # Encoding resume (returns a 2D array, take the first element)
    criteria_embeddings = model.encode(processed_criteria)  # Encoding all criteria (returns a 2D array)
    
    # Calculate similarity scores
    similarity_scores = {}
    print(Fore.MAGENTA + "Calculating similarity scores for each criterion:" + Style.RESET_ALL)
    for i, criterion in enumerate(criteria_list):
        criterion_embedding = criteria_embeddings[i]
        similarity = cosine_similarity([resume_embedding], [criterion_embedding])[0][0]
        # Scale similarity to 0-5 range
        scaled_score = round(similarity * 5, 2)
        # print(f"Similarity score for {criterion}: {scaled_score}")
        print(Fore.BLUE + "Scaled " + Fore.YELLOW + "Similarity score for " + Style.RESET_ALL + Fore.CYAN + criterion + Style.RESET_ALL + Fore.BLUE + " : " + Style.RESET_ALL + Fore.GREEN + str(scaled_score) + Style.RESET_ALL)
        similarity_scores[f"{criterion} (Similarity)"] = scaled_score
    
    return similarity_scores



def score_resume_with_llm(text: str, criteria: list) -> dict:
    print(Fore.BLUE + "Scoring resume with gpt-4o for each criterion" + Style.RESET_ALL)
    try:
        prompt = f"""You are an expert HR professional evaluating resumes against specific criteria. 

        **Instructions:**
        - Score each criterion strictly between `0` and `5` (0 = Not Mentioned, 5 = Perfect Match).
        - Deduct points if experience is vague or missing.
        - Return only a valid JSON object with exact key names.

        **Criteria:**
        {json.dumps(criteria, indent=2)}

        **Resume:**
        {text}

        **Example Response (Strict JSON format, no extra text):**
        {{
            "Bachelor's or Master's degree in Computer Science, AI, or related field": 5,
            "3+ years of experience with Python programming for AI/ML applications": 4,
            ...
        }}
        """

        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert HR professional. Strictly return JSON format scores."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Stricter scoring
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        # Extract response content
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()  # Remove markdown formatting

        # Parse JSON
        scores = json.loads(response_text)

        # Validate score range
        for key, value in scores.items():
            if not isinstance(value, (int, float)) or value < 0 or value > 5:
                scores[key] = 0  # Fix invalid values

        return scores

    except Exception as e:
        print(f"Error in score_resume_with_llm: {str(e)}")
        return {}

def score_resume(text: str, criteria: List[str]) -> dict:
    # Only use LLM scoring now
    return score_resume_with_llm(text, criteria)

def extract_candidate_name(text: str) -> str:
    # print("text to extract name from : ", text[:100])
    print(Fore.GREEN + "Extracting candidate name using LLM prompting and first 250 characters of resume" + Style.RESET_ALL)
    try:
        # First try with 250 characters
        truncated_text = text[:250]
        prompt = f"""
        Given this text from the start of a resume, what is the candidate's full name? 
        Return ONLY the name, with no additional text or explanation.
        If no name is found, return "Unknown Candidate".

        Text: {truncated_text}
        """
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a resume parser that extracts candidate names. Respond only with the name found."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )

        name = response.choices[0].message.content.strip()
        
        # # If no name found, try with 300 characters
        # if name == "Unknown Candidate":
        #     truncated_text = text[:300]
        #     prompt = f"""
        #     Given this text from the start of a resume, what is the candidate's full name? 
        #     Return ONLY the name, with no additional text or explanation.
        #     If no name is found, return "Unknown Candidate".

        #     Text: {truncated_text}
        #     """
        #     print("NAME prompt 2 :      ", prompt)
        #     response = client.chat.completions.create(
        #         model=OPENAI_DEPLOYMENT_NAME,
        #         messages=[
        #             {"role": "system", "content": "You are a resume parser that extracts candidate names. Respond only with the name found."},
        #             {"role": "user", "content": prompt}
        #         ],
        #         temperature=0,
        #         max_tokens=50
        #     )
            
        #     name = response.choices[0].message.content.strip()

        return name if name else "Unknown Candidate"

    except Exception as e:
        print(f"Error extracting name: {str(e)}")
        return "Unknown Candidate"

@app.post("/score-resumes", 
    summary="Score resumes against criteria",
    response_description="Excel file containing resume scores")
async def score_resumes(
    files: List[UploadFile] = File(..., description="PDF or DOCX resume files to analyze"),
    criteria: str = Form(..., description="JSON array of criteria to score against. Example: [{\"criterion\": \"Python experience\"}]")
):
    """
    Upload resumes and get them scored against your criteria.

    - **files**: One or more resume files (PDF/DOCX)
    - **criteria**: JSON string containing scoring criteria
    """
    criteria_list = json.loads(criteria)
    results = []

    for index, file in enumerate(files):
        content = await file.read()
        # print(f"{Fore.GREEN}Processing file {index + 1} of {len(files)} files: Name - " + Style.RESET_ALL + Fore.CYAN + file.filename + Style.RESET_ALL)
        print(f"{Style.BRIGHT + Fore.MAGENTA}Processing file {index + 1} of {len(files)} files: Name - " + Style.RESET_ALL + Fore.CYAN + file.filename + Style.RESET_ALL)
        try:
            if file.filename.endswith('.pdf'):
                text = await extract_text_from_pdf(content)
            elif file.filename.endswith('.docx'):
                text = extract_text_from_docx(content)
            else:
                continue
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting text from {file.filename}: {str(e)}")

        candidate_name = extract_candidate_name(text)
        
        # Get both LLM scores and similarity scores
        llm_scores = score_resume(text, criteria_list)
        similarity_scores = calculate_similarity_scores(text, criteria_list)
        print(Fore.GREEN + "Calculating total LLM and similarity scores" + Style.RESET_ALL)
        # print((Fore.BLUE + "Weighing LLM and embeddings similarity in 2:1 ratio" + Style.RESET_ALL))
        print(Fore.BLUE + "Weighing LLM and embeddings similarity in " + Fore.WHITE + "2:1" + Fore.BLUE + " ratio" + Style.RESET_ALL)
        # Combine all scores
        combined_scores = {
            "Candidate Name": candidate_name,
            **llm_scores,
            **similarity_scores,
            "Total LLM Score": sum(llm_scores.values()),
            "Total Similarity Score": sum(similarity_scores.values()),
            "Total Score": (2*sum(llm_scores.values())+sum(similarity_scores.values()))/3,
        }
        
        results.append(combined_scores)

    # Create DataFrame and save to Excel
    df = pd.DataFrame(results)
    print(Fore.GREEN + "Created Excel file with resume scores!" + Style.RESET_ALL)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        df.to_excel(tmp.name, index=False)
        tmp_path = tmp.name

    return FileResponse(
        tmp_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="resume_scores.xlsx"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 