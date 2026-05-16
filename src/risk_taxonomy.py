import os
import re
import csv
from typing import List
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
load_dotenv()
import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
# ==========================================
# 1. Define Risk Signals (18)
# ==========================================

class RiskSignal(str, Enum):
    S01_HOPELESSNESS_ESCALATION = "S01_hopelessness_escalation"
    S02_CATASTROPHIZING = "S02_catastrophizing"
    S03_ALL_OR_NOTHING_THINKING = "S03_all_or_nothing_thinking"
    S04_SELF_BLAME_AMPLIFICATION = "S04_self_blame_amplification"
    S05_EMOTIONAL_VOLATILITY = "S05_emotional_volatility"
    S06_SOCIAL_WITHDRAWAL_LANGUAGE = "S06_social_withdrawal_language"
    S07_HELPLESSNESS_LOSS_OF_AGENCY = "S07_helplessness_loss_of_agency"
    S08_NEGATIVE_SENTIMENT_TREND = "S08_negative_sentiment_trend"
    S09_RUMINATION_PATTERNS = "S09_rumination_patterns"
    S10_EMOTIONAL_NUMBING = "S10_emotional_numbing"
    S11_COGNITIVE_DISTORTION_DENSITY = "S11_cognitive_distortion_density"
    S12_ESCALATION_RISK_COMPOSITE = "S12_escalation_risk_composite"
    S13_SUICIDAL_IDEATION = "S13_suicidal_ideation"
    S14_SUBSTANCE_ABUSE = "S14_substance_abuse"
    S15_SELF_HARM = "S15_self_harm"
    S16_TRAUMA_OR_ABUSE = "S16_trauma_or_abuse"
    S17_MANIC_OR_HYPOMANIC_EPISODES = "S17_manic_or_hypomanic_episodes"
    S18_PSYCHOTIC_SYMPTOMS = "S18_psychotic_symptoms"


class ClassificationOutput(BaseModel):
    labels: List[RiskSignal] = Field(default_factory=list)


# ==========================================
# 2. LLM Setup
# ==========================================

# Build valid label list dynamically from enum so prompt stays in sync
VALID_LABELS = "\n".join([f"- {signal.value}" for signal in RiskSignal])

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

structured_llm = llm.with_structured_output(ClassificationOutput)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""
You are an expert clinical-text annotation assistant labeling therapy transcripts for linguistic and emotional distress signals.

You will receive a full therapy transcript containing both therapist and patient dialogue.

Your task:
- Use the FULL transcript only to understand context.
- Assign risk signal labels ONLY based on what the PATIENT says.
- Do NOT assign any label based on therapist questions, therapist summaries, therapist interpretations, or therapist suggestions.
- If a therapist mentions a symptom but the patient does not express or confirm it, do NOT label it.
- If the patient explicitly confirms, repeats, agrees with, or elaborates on a therapist's statement, then you may label based on the patient's confirmation.
- Do NOT diagnose the patient.
- Do NOT infer beyond the text.
- Prefer precision over recall: only label signals with clear or moderate evidence in patient language.

Important speaker rule:
- Patient evidence includes only lines spoken by the patient/client.
- Therapist evidence is context only and must never be the direct basis for a label.

Return all applicable labels from the predefined list below.
If no patient-language evidence supports any signal, return an empty list.

You MUST only use labels exactly as listed below, no variations:

{VALID_LABELS}
        """),
        ("human", """
Full Therapy Transcript:
{text}

Classify the applicable risk signals using only patient dialogue as evidence.
""")
    ]
)

chain = prompt | structured_llm


# ==========================================
# 3. Helper: Extract Patient Text
# ==========================================

def extract_patient_text(full_text: str) -> str:
    # Fix: use re.split to handle single or double newlines between speaker turns
    pattern = r"<Patient>:\s*(.*?)(?=\n{1,2}<|$)"
    matches = re.findall(pattern, full_text, re.DOTALL)
    return "\n".join([m.strip() for m in matches if m.strip()])


# ==========================================
# 4. Process Train Folder
# ==========================================

def process_train_folder(train_path: str, output_csv: str):
    rows = []
    txt_files = [f for f in os.listdir(train_path) if f.endswith(".txt")]

    for filename in tqdm(txt_files):
        if not filename.endswith(".txt"):
            continue

        transcript_id = os.path.splitext(filename)[0]
        file_path = os.path.join(train_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        #patient_text = extract_patient_text(full_text)
        patient_text = full_text

        if not patient_text.strip():
            risk_string = ""
        else:
            result = chain.invoke({"text": patient_text})
            risk_string = ",".join([label.value for label in result.labels])

        rows.append({
            "transcript_id": transcript_id,
            "text": full_text.replace("\n", " "),
            "risk_signals": risk_string
        })

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["transcript_id", "text", "risk_signals"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows, columns=["transcript_id", "text", "risk_signals"])

    print(f"\nSaved to {output_csv}")


# ==========================================
# 5. Run
# ==========================================

if __name__ == "__main__":
    FOLDER = r"C:\Users\samee\Documents\CHIEAC\HOPE_WSDM_2022\Train"
    process_train_folder(
        train_path=FOLDER,
        output_csv="src/train_risk_signals.csv"
    )
