from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from loguru import logger


class GuardRails:

    def __init__(self, path = "jackhhao/jailbreak-classifier") -> None:
    
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

    def classify_prompt(self, prompt):
        # Encode the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Get classification logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Extract label with highest probability
        predicted_class = torch.argmax(probabilities).item()
        logger.info(f"Prompt classified as: {predicted_class}")
        return predicted_class


# 0 -> bening
# 1 -> Jailbreak

