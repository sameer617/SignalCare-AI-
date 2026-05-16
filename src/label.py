import json
from typing import List, Tuple
from risk_taxonomy import ClassificationOutput, chain

def label_utterances(utterances: List[Tuple[str, int, str]], output_file: str):
    with open(output_file, "w") as f:
        for transcript_id, patient_turn_id, redacted_text in utterances:
            result = chain.invoke({"text": redacted_text})
            
            distress_signals = [label.value for label in result.labels]
            signal_confidence = {label.value: 0.8 for label in result.labels} 
            
            output = {
                "transcript_id": transcript_id,
                "patient_turn_id": patient_turn_id, 
                "text": redacted_text,
                "sentiment_score": 0.0, 
                "sentiment_label": "neutral",
                "primary_emotion": "neutral",
                "emotion_intensity": 1,
                "distress_signals": distress_signals,
                "signal_confidence": signal_confidence,
                "cognitive_distortions": []
            }
            
            f.write(json.dumps(output) + "\n")


if __name__ == "__main__":
    utterances = []
    for line in open("stdout.txt"):
        transcript_id, patient_turn_id, redacted_text = line.strip().split(",", maxsplit=2)
        utterances.append((transcript_id, int(patient_turn_id), redacted_text))
        
    label_utterances(utterances, "train_utterance_labels.jsonl")
