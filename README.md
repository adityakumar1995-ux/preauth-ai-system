# PreAuth.ai — AI-Powered Prior Authorization Documentation Review Copilot

PreAuth.ai is a Streamlit-based AI application built for the **Harnessing AI for Business** final project. The product helps prior authorization teams review synthetic or de-identified prior authorization packets before payer submission. It identifies documentation gaps, denial-risk factors, recommended fixes, and relevant payer-policy sources.

This project is designed as a **business-facing AI product**, not just a technical demo. It includes a landing page, an integrated AI workflow, model evaluation, business impact metrics, and responsible AI guardrails.

---

## 1. Business Problem

Prior authorization is a common healthcare workflow where providers must obtain payer approval before certain procedures, medications, or services are covered. These requests are often delayed or denied when documentation is incomplete, unclear, outdated, or misaligned with payer requirements.

The main problem PreAuth.ai addresses is:

> Prior authorization teams often do not know whether a packet is likely to be denied until after submission.

This creates:

- Administrative rework
- Delayed patient access to care
- Higher operational cost
- Provider frustration
- Avoidable denials
- Patient safety and continuity-of-care concerns

PreAuth.ai moves the documentation review earlier in the workflow by helping staff identify likely denial risks before submission.

---

## 2. Product Summary

PreAuth.ai allows a user to upload a synthetic or de-identified prior authorization PDF packet. The app extracts relevant text and fields, retrieves payer-policy context from a local knowledge base, and uses an AI review model to generate a structured documentation risk assessment.

The output includes:

- Risk score from 0 to 100
- Risk label: Low, Medium, or High
- Denial-risk reasons
- Missing documentation
- Recommended fixes
- Retrieved policy sources
- Downloadable PDF report
- Model evaluation metrics
- Business impact estimates

The app is intended for **decision support only**. It does not approve or deny care, submit claims, replace payer review, or provide medical/legal advice.

---

## 3. Target Users

### Primary Users

- Prior authorization specialists
- Provider office staff
- Revenue cycle management teams
- Healthcare operations teams

### Secondary Stakeholders

- Physicians and clinical reviewers
- Patients waiting for authorization
- Healthcare administrators
- Compliance and quality teams
- Payers receiving more complete documentation packets

---

## 4. Core User Workflow

1. Open the Streamlit app.
2. Review the **Product Overview** landing page.
3. Go to **New Submission**.
4. Upload a synthetic or de-identified prior authorization PDF packet.
5. Extract fields from the packet.
6. Review and edit the extracted fields if needed.
7. Select **Groq AI Review Mode**.
8. Run the prior authorization analysis.
9. Review the risk score, denial reasons, missing documentation, recommendations, and policy sources.
10. Download the PDF report if needed.
11. Use **Test Evaluation** to compare model predictions against expected risk labels.
12. Use **Model Performance** to review confusion matrix, precision, recall, and business impact metrics.

---

## 5. Key Features

### 5.1 Product Overview Landing Page

The landing page explains:

- Business problem
- Target users
- AI solution
- Value proposition
- Product features
- Responsible AI positioning
- Suggested demo flow

This page makes the product understandable to a business user or project evaluator before they interact with the technical workflow.

### 5.2 New Submission Page

The New Submission page allows users to:

- Upload a prior authorization PDF packet
- Extract text and structured fields
- Review/edit patient and request information
- Run AI-based documentation risk analysis
- View output in a business-readable format
- Export the analysis as a PDF report

### 5.3 AI Review Modes

The system currently supports three analysis modes.

#### Groq AI Review Mode

This is the default AI review mode. It uses a Groq-hosted LLM through an OpenAI-compatible API interface.

The model receives:

- Extracted request fields
- Full uploaded PDF packet text
- Retrieved payer-policy context

It returns structured JSON containing risk score, risk level, denial reasons, missing documentation, and recommended fixes.

#### Gemini Live Mode

Gemini Live Mode is a backup AI review option. It uses Gemini with the same extracted packet text and retrieved policy context.

#### Local Fallback Mode

Local Fallback Mode is used only when live AI review is unavailable because of API errors, rate limits, missing keys, or connectivity issues.

This fallback is not the primary model. It exists to keep the app usable during demos and to avoid complete workflow failure.

### 5.4 Policy Retrieval

The app uses a local payer-policy knowledge base. Retrieved context is passed into the AI prompt and displayed in the final output.

This improves explainability because users can see which policy sources were retrieved during the review.

### 5.5 Test Evaluation Page

The Test Evaluation page allows labeled testing on synthetic records.

For each record, the user enters the expected risk class:

- Low
- Medium
- High

The model then generates its own risk prediction. The app stores the expected and predicted labels and updates performance metrics.

The page generates:

- 3×3 confusion matrix
- Class-level precision
- Class-level recall
- Weighted precision
- Weighted recall
- Risk-label accuracy
- Business impact estimates

### 5.6 Model Performance Page

The Model Performance page summarizes:

- Total evaluation records
- Accuracy
- Weighted precision
- Weighted recall
- Confusion matrix
- Class-level precision/recall
- Evaluation criteria
- Limitations

### 5.7 PDF Report Export

The app can generate a downloadable PDF report for a prior authorization review. This report summarizes the risk score, denial reasons, missing documentation, recommended fixes, and source information.

---

## 6. System Architecture

The final system follows this architecture:

```text
User
 |
 | uploads synthetic/de-identified prior authorization PDF
 v
Streamlit UI
 |
 | sends uploaded PDF to extractor
 v
PDF Extraction Layer
 |
 | extracts raw packet text and structured fields
 v
PreAuth Analysis Engine
 |
 | enriches weak extracted fields using raw PDF text
 | retrieves relevant payer-policy context
 v
Policy Retrieval Layer / Local Knowledge Base
 |
 | returns relevant policy chunks and metadata
 v
AI Review Layer
 |
 | Groq AI Review Mode
 | Gemini Live Mode backup
 | Local Fallback Mode if APIs fail
 v
Structured JSON Output
 |
 | validated and displayed in UI
 v
Business-Facing Output
 |
 | risk score
 | risk label
 | denial reasons
 | missing documentation
 | recommended fixes
 | retrieved policy sources
 | downloadable PDF report
 | evaluation metrics
```

---

## 7. Codebase Structure

```text
preauth_ai_system/
│
├── app.py
├── analysis_engine.py
├── rag_engine.py
├── pdf_extractor.py
├── report_generator.py
├── ingest_knowledge_base.py
├── evaluate_synthetic_records.py
├── test_groq.py
├── test_gemini.py
├── test_analysis_engine.py
├── test_pdf_extractor.py
├── test_rag_engine.py
├── test_retrieval.py
├── test_setup.py
├── requirements.txt
├── .gitignore
│
├── data/
├── knowledge_base/
├── sample_cases/
├── evaluation_records/
├── vector_store/
└── __pycache__/
```

---

## 8. Important Files

### `app.py`

Main Streamlit application.

Responsibilities:

- Defines page layout and navigation
- Displays Product Overview, Dashboard, New Submission, Patient Lookup, Test Evaluation, Model Performance, and How It Works pages
- Handles PDF upload and field review
- Calls the AI analysis engine
- Displays model output
- Stores submission and evaluation records locally
- Generates confusion matrix and business metrics
- Provides PDF report download buttons

### `analysis_engine.py`

Main AI review engine.

Responsibilities:

- Loads API keys from `.env` or Streamlit secrets
- Enriches extracted patient/request fields using raw PDF text
- Retrieves policy context
- Builds the LLM review prompt
- Calls Groq AI Review Mode
- Calls Gemini Live Mode as backup
- Falls back to local fallback logic if APIs fail
- Validates AI JSON output
- Returns structured risk assessment to the app

### `rag_engine.py`

Policy retrieval logic.

Responsibilities:

- Searches the local vector store / knowledge base
- Retrieves relevant payer-policy context
- Formats retrieved context for the AI prompt
- Returns source metadata for transparency

### `pdf_extractor.py`

PDF extraction module.

Responsibilities:

- Extracts raw text from uploaded PDFs
- Attempts to parse common prior authorization fields
- Returns extracted fields and raw packet text to the app

### `report_generator.py`

PDF report generation module.

Responsibilities:

- Converts the AI analysis result into a downloadable report
- Includes patient/request fields, risk score, denial reasons, missing documentation, recommendations, and sources

### `ingest_knowledge_base.py`

Knowledge base ingestion script.

Responsibilities:

- Reads payer-policy documents
- Splits documents into chunks
- Creates or updates the local vector store
- Prepares policy documents for retrieval

### `test_groq.py`

Simple test script to verify that the Groq API key and model are working.

### `requirements.txt`

Lists Python dependencies required to run the project.

### `.gitignore`

Prevents sensitive or unnecessary files from being committed to GitHub, such as:

- `.env`
- `__pycache__/`
- local data files
- large vector store files if excluded
- temporary outputs

---

## 9. AI Prompting Approach

The project uses a structured generative AI prompt rather than an open-ended chatbot prompt.

The model is instructed to behave as a prior authorization documentation reviewer. The prompt includes:

- Role definition
- Business task
- Extracted request fields
- Full uploaded packet text
- Retrieved payer-policy context
- Risk scoring rubric
- Specialty-awareness instructions
- Hallucination controls
- JSON-only output requirements

### Short System Instruction

The model receives a short system-level instruction:

```text
You are a careful prior authorization documentation reviewer. Return only valid JSON.
```

### Main Review Instructions

The main prompt tells the model to:

- Read the full packet before deciding anything is missing
- Infer the procedure or service category
- Avoid applying one specialty’s documentation rules to another specialty
- Use service-specific checklist evidence when available
- Avoid inventing facts
- Avoid external verification of IDs or eligibility
- Use retrieved policy context without forcing irrelevant requirements
- Assess documentation risk only
- Return structured JSON

### Risk Score Mapping

```text
0–39   = Low Risk
40–69  = Medium Risk
70–100 = High Risk
```

### Required JSON Output

```json
{
  "risk_score": 0,
  "risk_level": "low",
  "summary": "brief business-facing summary",
  "denial_reasons": ["specific reason 1", "specific reason 2"],
  "missing_documentation": ["specific missing item 1", "specific missing item 2"],
  "recommended_fixes": ["specific actionable fix 1", "specific actionable fix 2"]
}
```

---

## 10. Evaluation Design

The system includes an in-app model evaluation workflow.

Each synthetic test record has an expected label:

- Low risk
- Medium risk
- High risk

The model generates a predicted label.

The app compares expected vs. predicted labels and updates a 3×3 confusion matrix.

### Confusion Matrix Layout

```text
Rows    = Expected risk category
Columns = Model-generated risk category
```

Example:

```text
Expected \ Predicted | Low | Medium | High
Low                  |     |        |
Medium               |     |        |
High                 |     |        |
```

### Class-Level Precision

Precision measures how often the model is correct when it predicts a class.

```text
Precision = True Positives / Total Predicted as That Class
```

### Class-Level Recall

Recall measures how often the model catches actual cases of a class.

```text
Recall = True Positives / Total Actual Cases in That Class
```

### Weighted Precision

Weighted precision averages class-level precision scores by the number of actual examples in each class.

### Weighted Recall

Weighted recall averages class-level recall scores by the number of actual examples in each class.

### Risk-Label Accuracy

```text
Accuracy = Correct Predictions / Total Evaluation Records
```

---

## 11. Business Metrics

The project connects model performance to business impact. This is important because the goal is not only to build an AI tool, but to show how it can affect business outcomes.

The business metrics are calculated using user-entered assumptions and model evaluation results.

### Input Assumptions

The Test Evaluation page allows users to enter:

- Prior authorizations completed per week
- Current staff time per prior authorization review
- Current denial rate
- Current adverse patient event / delay rate due to PA review
- Assumed maximum time reduction when the model is reliable
- Share of denials assumed preventable through better documentation
- Share of adverse events or delays assumed preventable through better review

### Estimated Time Saved per Prior Authorization

```text
Estimated Time Saved per PA =
Current Staff Review Time × Maximum Time Reduction % × Weighted Precision
```

### Estimated Weekly Time Saved

```text
Estimated Weekly Time Saved =
(Prior Authorizations per Week × Estimated Time Saved per PA) / 60
```

### Projected Denial Rate

```text
Denial Rate Reduction =
Current Denial Rate × Preventable Denial Share × Weighted Recall

Projected Denial Rate =
Current Denial Rate - Denial Rate Reduction
```

### Projected Adverse Event / Delay Rate

```text
Adverse Event Reduction =
Current Adverse Event / Delay Rate × Preventable Adverse Event Share × Weighted Recall

Projected Adverse Event / Delay Rate =
Current Adverse Event / Delay Rate - Adverse Event Reduction
```

These business metrics are directional estimates, not validated operational guarantees.

---

## 12. Setup Instructions

### 12.1 Clone the Repository

```bash
git clone https://github.com/your-username/preauth-ai-system.git
cd preauth-ai-system
```

Replace the repository URL with the actual GitHub repository URL.

### 12.2 Create a Virtual Environment

```bash
python -m venv venv
```

### 12.3 Activate the Virtual Environment

#### Windows PowerShell

```powershell
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 12.4 Install Dependencies

```bash
pip install -r requirements.txt
```

If using Groq, make sure the `openai` package is included because Groq uses an OpenAI-compatible API.

```text
openai
```

---

## 13. Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

GEMINI_API_KEY=your_gemini_api_key_here
```

Only `GROQ_API_KEY` is required for the default mode.

`GEMINI_API_KEY` is optional and used as a backup.

Do not commit `.env` to GitHub.

---

## 14. Testing API Access

### Test Groq

```bash
python test_groq.py
```

Expected output:

```text
GROQ_API_KEY found: True
GROQ_MODEL: llama-3.3-70b-versatile
{"status":"groq working"}
```

If this fails, check:

- API key is correct
- Key has not been revoked
- `.env` is in the project root
- `openai` package is installed
- Groq model name is correct
- Rate limit has not been reached

### Test Gemini

```bash
python test_gemini.py
```

This is optional because Gemini is used as a backup.

---

## 15. Run the Application

```bash
python -m streamlit run app.py
```

The app should open in the browser.

If it does not open automatically, copy the local URL from the terminal and paste it into the browser.

---

## 16. Recommended Demo Flow

1. Open the app.
2. Start on **Product Overview**.
3. Go to **New Submission**.
4. Upload a synthetic prior authorization PDF.
5. Click **Extract Fields from PDF**.
6. Review the extracted fields.
7. Select **Groq AI Review Mode**.
8. Click **Run Prior Authorization Analysis**.
9. Review risk score, risk label, denial reasons, missing documentation, recommended fixes, and retrieved policy sources.
10. Download the PDF report.
11. Go to **Test Evaluation**.
12. Upload labeled synthetic test records one by one.
13. Review the confusion matrix, precision, recall, and business metrics.
14. Go to **Model Performance**.
15. Show summary metrics and evaluation criteria.

---

## 17. Streamlit Cloud Deployment

### 17.1 Push Code to GitHub

```bash
git add .
git commit -m "Finalize PreAuth AI app"
git push
```

### 17.2 Deploy on Streamlit Cloud

1. Go to Streamlit Cloud.
2. Select **Deploy app from GitHub**.
3. Choose the repository.
4. Set the main file path:

```text
app.py
```

5. Set Python version to a compatible version such as Python 3.11.
6. Add secrets.

### 17.3 Streamlit Secrets

Add secrets in Streamlit Cloud:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"

GEMINI_API_KEY = "your_gemini_api_key_here"
```

Do not put API keys directly in the code.

---

## 18. Troubleshooting

### App falls back to `fallback_local_rules`

This means Groq and Gemini were unavailable.

Check:

- `GROQ_API_KEY` is present
- Key is valid
- `openai` package is installed
- Groq rate limit has not been reached
- `GROQ_MODEL` is available
- Streamlit secrets were set correctly in the cloud

### Groq API key was shown in a screenshot

Immediately revoke the exposed key in Groq Console and create a new one.

### App says Gemini quota exceeded

Use Groq AI Review Mode or wait for Gemini quota reset.

### PDF extraction looks incomplete

Open the extracted raw text expander and verify whether the PDF text is readable.

If the raw text is empty or messy, the PDF may be scanned/image-based and may require OCR, which is not part of the current prototype.

### Wrong risk output

Check the backend mode displayed in the result.

If it says:

```text
groq_ai_review
```

the AI model generated the result.

If it says:

```text
fallback_local_rules
```

the live AI did not run, and the fallback result should not be treated as the main model output.

---

## 19. Responsible AI and Guardrails

PreAuth.ai includes several guardrails:

- Synthetic or de-identified data only
- Clear academic prototype disclaimer
- Human review required
- AI output is decision support only
- No automatic approval or denial
- No medical diagnosis or treatment recommendation
- Retrieved policy sources shown for transparency
- Structured JSON output validation
- Fallback behavior when live APIs fail
- Evaluation page for model performance monitoring

---

## 20. Privacy and Security Notes

This prototype is not HIPAA-compliant and should not be used with real patient data.

A production version would require:

- HIPAA-compliant infrastructure
- Access controls
- Encryption at rest and in transit
- Audit logs
- Secure API key management
- Business Associate Agreements where required
- Legal and compliance review
- Clinical validation
- Integration with EHR/revenue cycle systems

---

## 21. Limitations

Current limitations:

- Uses synthetic/de-identified records only
- No historical payer approval/denial dataset
- Risk score is an AI-generated documentation assessment, not a validated payer decision prediction
- Evaluation is based on synthetic labeled packets
- Payer policies may change over time
- Groq/Gemini availability and rate limits may affect live review
- Local fallback is not the primary model
- The app does not perform OCR for scanned documents
- The app does not submit claims or connect to payer portals
- The app is not production-ready for clinical use

---

## 22. Future Improvements

Potential future enhancements:

- Add more payer policies and plan types
- Add automatic OCR for scanned packets
- Add reviewer feedback loop
- Add audit logging
- Add user authentication and role-based access
- Add HIPAA-compliant cloud deployment
- Add EHR/revenue cycle system integration
- Add payer-specific benchmarking
- Add continuous model monitoring
- Add subgroup fairness audits
- Add automatic policy update monitoring
- Add more robust source citations to exact policy excerpts

---

## 23. GitHub Submission Notes

Before submitting:

- Confirm `.env` is not committed
- Confirm API keys are not visible in code
- Confirm app runs locally
- Confirm Streamlit app runs if deployed
- Confirm README is included
- Confirm final report includes screenshots and metrics
- Confirm evaluation results are downloaded if needed
- Confirm synthetic test records are not sensitive

---

## 24. Academic Disclaimer

This application was built for academic demonstration as part of a business AI final project. It is not intended for real patient-care decisions, actual payer submissions, medical advice, legal advice, or production healthcare operations.

---

## 25. Author

Aditya Kumar  
Rohith Thappa
Cole Weston Dwiggins
Lucca Diana
Nishat Chowdhury
University of Maryland, College Park  
Harnessing AI for Business  
PreAuth.ai Final Project
