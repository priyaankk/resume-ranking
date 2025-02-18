Here is your README.md file:

# Job Screening APIs

This repository contains two APIs for automating job screening:

1. **Job Criteria Extractor API** – Extracts key hiring criteria from job description PDFs.
2. **Resume Ranking API** – Ranks multiple resumes based on extracted criteria and generates an Excel report.

## Setup Instructions

Follow these steps to set up and run the project:

1. **Configure Environment Variables**  
   - Fill the `.env` file with your **Azure credentials**.  
   - Ensure the necessary deployments are created on the Azure portal.

2. **Create a Virtual Environment**  
   ```sh
   python -m venv venv

	3.	Activate the Virtual Environment
	•	Windows:

venv\Scripts\activate


	•	Mac/Linux:

source venv/bin/activate


	4.	Install Dependencies

pip install -r requirements.txt


	5.	Run the API Server

uvicorn main:app --host 0.0.0.0 --port 8000 --reload


	6.	Access API Documentation
Open your browser and go to:

http://localhost:8000/docs



License

This project is licensed under MIT License.

