import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple

FOLDER = r"C:\Users\samee\Documents\CHIEAC\HOPE_WSDM_2022\Train"

def list_files(FOLDER):
    """
    Retrieve all `.txt` files from the specified directory.

    Args:
        FOLDER (str): Path to the directory containing text files.

    Returns:
        list[str]: A list of file paths matching the `.txt` extension.
    """
    files = glob.glob(FOLDER + "/*.txt")
    return files


def read_files(file_path: str) -> str:
    """
    Read and return the full text content of a file.

    Args:
        file_path (str): Path to the file to be read.

    Returns:
        str: The complete content of the file as a string.
    """
    with open(file_path) as f:
        text = f.read()

    return text


def load_hugging_face_model(model_name: str):
    pass


def sentence_level_sentiment(model_name: str, text: str) -> Tuple[str, float]:
    """
    Predict the sentiment (or classification label) of a given text using a
    Hugging Face sequence classification model.

    This function loads the specified model and tokenizer from the Hugging Face Hub,
    performs inference on the input text (then caches them), and returns the predicted label along with
    its confidence score.

    Args:
        model_name (str): Name or path of the Hugging Face model to load.
            Example: "paulagarciaserrano/roberta-depression-detection"
        text (str): Input sentence or text to classify.

    Returns:
        Tuple[str, float]:
            - label (str): Predicted class label (e.g., "positive", "negative", "depression").
            - score (float): Confidence score (probability) associated with the predicted label.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_class_id = torch.max(probs, dim=-1)

    label = model.config.id2label[predicted_class_id.item()]
    score = confidence.item()

    return label, score

def sentence_level_emotions(model_name: str, text: str, threshold: float = 0.5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0]

    emotion_scores = {
        model.config.id2label[i]: probs[i].item()
        for i in range(len(probs))
    }

    active_emotions = [
        label for label, score in emotion_scores.items()
        if score >= threshold
    ]

    return emotion_scores, active_emotions