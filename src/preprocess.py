"""
preprocess.py
-------------
Stage 1 of the SignalCare AI pipeline.

For each transcript in the input directory:
  1. Splits the transcript into individual patient utterances
  2. Runs Presidio PII redaction on each utterance
  3. Saves a redacted copy of each transcript to output_dir/redacted/
  4. Saves all utterances as a JSONL file: output_dir/utterances.jsonl

Each line in utterances.jsonl has the schema:
  {
    "transcript_id": "Transcript_1",
    "patient_turn_id": 1,
    "text": "<redacted utterance text>"
  }

Usage:
  # Run all splits (Train, Validation, Test):
  python src/preprocess.py

  # Run a specific split:
  python src/preprocess.py --split Validation
  python src/preprocess.py --split Test
  python src/preprocess.py --split Train
"""

import os
import re
import json
import argparse
from typing import List, Dict
from tqdm import tqdm
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOPE_DIR   = os.path.join(REPO_ROOT, "HOPE_WSDM_2022")
ALL_SPLITS = ["Train", "Validation", "Test"]


# ==========================================
# 1. Parse patient utterances from transcript
# ==========================================

def extract_patient_utterances(transcript: str) -> List[str]:
    """
    Splits a raw transcript into individual patient utterances.
    Handles both single and double newlines between speaker turns.
    Correctly captures the last patient turn even if not followed by <Therapist>.
    """
    # Split on any speaker tag: <Patient>: or <Therapist>:
    # Keep track of which speaker each segment belongs to
    segments = re.split(r'(<Patient>:|<Therapist>:)', transcript)

    utterances = []
    current_speaker = None

    for segment in segments:
        segment = segment.strip()
        if segment == "<Patient>:":
            current_speaker = "patient"
        elif segment == "<Therapist>:":
            current_speaker = "therapist"
        elif segment and current_speaker == "patient":
            utterances.append(segment)

    return utterances


# ==========================================
# 2. PII Redaction
# ==========================================

def redact_pii(text: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine) -> str:
    """
    Runs Presidio PII detection and anonymization on a single text string.
    Returns the anonymized text.
    """
    results = analyzer.analyze(text, language="en")
    if not results:
        return text
    return anonymizer.anonymize(text, results).text


# ==========================================
# 3. Main preprocessing pipeline
# ==========================================

def preprocess_transcripts(input_dir: str, output_dir: str) -> List[Dict]:
    """
    Processes all transcripts in input_dir.
    Saves:
      - Redacted transcript copies to output_dir/redacted/
      - All utterances to output_dir/utterances.jsonl

    Returns a list of utterance dicts with keys:
      transcript_id, patient_turn_id, text
    """
    redacted_dir = os.path.join(output_dir, "redacted")
    os.makedirs(redacted_dir, exist_ok=True)

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    all_utterances = []

    txt_files = sorted([
        f for f in os.listdir(input_dir)
        if f.startswith("Transcript") and f.endswith(".txt")
    ])

    print(f"\nFound {len(txt_files)} transcripts in {input_dir}")

    for filename in tqdm(txt_files, desc="Preprocessing transcripts"):
        transcript_id = os.path.splitext(filename)[0]
        filepath = os.path.join(input_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Step 1: Extract patient utterances
        utterances = extract_patient_utterances(raw_text)

        if not utterances:
            print(f"  [WARN] No patient utterances found in {filename}, skipping.")
            continue

        # Step 2: PII redact each utterance
        redacted_utterances = []
        for turn_id, utterance in enumerate(utterances, start=1):
            redacted = redact_pii(utterance, analyzer, anonymizer)
            redacted_utterances.append(redacted)
            all_utterances.append({
                "transcript_id": transcript_id,
                "patient_turn_id": turn_id,
                "text": redacted
            })

        # Step 3: Save redacted transcript copy
        redacted_filename = f"{transcript_id}_Redacted.txt"
        with open(os.path.join(redacted_dir, redacted_filename), "w", encoding="utf-8") as f:
            for i, utt in enumerate(redacted_utterances, start=1):
                f.write(f"[Turn {i}] {utt}\n\n")

    # Step 4: Save all utterances to JSONL
    jsonl_path = os.path.join(output_dir, "utterances.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for utt in all_utterances:
            f.write(json.dumps(utt) + "\n")

    print(f"\nDone.")
    print(f"  Total utterances extracted : {len(all_utterances)}")
    print(f"  Redacted transcripts saved : {redacted_dir}")
    print(f"  Utterances JSONL saved     : {jsonl_path}")

    return all_utterances


# ==========================================
# 4. Run
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HOPE transcript splits.")
    parser.add_argument(
        "--split",
        choices=ALL_SPLITS,
        default=None,
        help="Which split to process. Omit to process all splits."
    )
    args = parser.parse_args()

    splits = [args.split] if args.split else ALL_SPLITS

    for split in splits:
        input_dir  = os.path.join(HOPE_DIR, split)
        output_dir = os.path.join(REPO_ROOT, "processed", split)
        print(f"\n{'='*50}")
        print(f"Processing split: {split}")
        print(f"{'='*50}")
        preprocess_transcripts(input_dir, output_dir)
