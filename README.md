# SherlockAI — AI Compliance Triage Operator (Demo)

SherlockAI is an AI-native compliance triage operator. It turns inbound tickets (chat/email/support cases) into **structured, evidence-grounded triage outputs**, drafts the **next best questions** and a **concise internal memo**, and then routes everything through a **human-only decision gate** with audit logging.

This repo is built to be **demo**:
- **Deterministic** routing + fact extraction for predictability
- **LLM** used only for the “cognitive” parts (questions + memo)
- **Safe fallbacks** so the demo never breaks

---

## What SherlockAI does

Given a case (ticket + metadata), SherlockAI produces:

1) **Triage**
- `risk_category` (e.g., account_takeover, fraud, privacy…)
- `priority` (P0–P3)
- `routing_queue` (ATO, Fraud, Privacy, …)
- `confidence`
- **Rationale evidence**: short quote spans with character offsets

2) **Facts**
- Structured fields (e.g., device change, unauthorized claim)
- Each fact includes **evidence spans** (or `unknown` if not supported)

3) **Missing-info Questions (LLM)**
- 3–5 questions to resolve ambiguity safely
- Each question includes:
  - `why` it matters
  - `cost` (cheap/medium/expensive)
  - `evidence_quote` grounded in the ticket text, OR `"missing"` if the info is absent

4) **Internal Memo (LLM)**
- Short summary, recommendation, risks
- Includes **evidence quotes** from the ticket text

5) **Human-only Gate**
- Explicitly blocks high-impact actions from being automated

6) **Audit Logging**
- Records AI recommendation vs human final decision (queue/priority + reviewer note)

---

## We keep Humans in the loop for critical decisions. 

Any **client-impact or regulatory-impact action**, especially:
- restricting/freezing an account
- stopping or reversing money movement
- escalating to regulatory reporting

These decisions require legal/compliance judgment, a broader context than a single ticket, and careful calibration of false positives/negatives.

---

## Demo flow

1) Select a case (recommend **C-0003** or **C-0002**) and click **Run triage**
2) Show **Triage** outputs + evidence span
3) Show **Facts** with evidence spans (`unknown` where appropriate)
4) Show **Questions** (LLM) + grounding quotes
5) Show **Memo** (LLM) + evidence quotes
6) Go to **Human Decision**:
   - approve/override
   - add a brief reviewer note
   - click **Save decision to audit log**
7) Show **Audit JSON** tab (and/or the saved confirmation)

---

## Setup

### 1) Create & activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
   
```bash
pip install -r requirements.txt
```

3) Configure environment variables

Edit .env: You can change the model and you need to enter your Hugging Face token
```bash
USE_LLM=1
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
HF_BASE_URL=https://router.huggingface.co/v1
HF_MODEL=Qwen/Qwen2.5-72B-Instruct
```

4) Run the app
```bash
streamlit run app.py
```
