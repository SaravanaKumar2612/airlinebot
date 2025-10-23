from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import csv
import os
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

app = FastAPI()

# Allow frontend access (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev, open CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data model ---
class Message(BaseModel):
    text: str

class Feedback(BaseModel):
    text: str
    predicted_intent: str
    correct: bool
    true_label: str = None  # Optional field for human agents to provide correct label

# --- Config ---
MODEL_PATH = "../dilbert_airline"

# Load the DistilBERT model and tokenizer
try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print(f"DistilBERT model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None
    tokenizer = None

# --- Retraining Classes and Functions ---
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def check_and_retrain_model():
    """Check if retraining is needed and trigger retraining"""
    try:
        # Read feedback data
        if not os.path.exists("feedback_log.csv"):
            return
        
        df = pd.read_csv("feedback_log.csv")
        
        # Count incorrect predictions
        incorrect_count = len(df[df['correct'] == False])
        
        print(f"Total incorrect predictions: {incorrect_count}")
        
        # Check if threshold reached (50 incorrect predictions)
        if incorrect_count >= 50:
            print("Retraining threshold reached. Starting model retraining...")
            retrain_model(df)
        else:
            print(f"Retraining threshold not reached. Need {50 - incorrect_count} more incorrect predictions.")
            
    except Exception as e:
        print(f"Error checking retraining threshold: {e}")

def retrain_model(feedback_df):
    """Retrain the model using feedback data"""
    try:
        # Filter data with true labels
        training_data = feedback_df[
            (feedback_df['correct'] == False) & 
            (feedback_df['true_label'].notna()) & 
            (feedback_df['true_label'] != '')
        ]
        
        if len(training_data) < 10:  # Need minimum samples for retraining
            print("Not enough labeled data for retraining. Need at least 10 samples.")
            return
        
        print(f"Retraining with {len(training_data)} samples")
        
        # Get unique labels and create label mapping
        unique_labels = sorted(training_data['true_label'].unique().tolist())
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        # Prepare training data
        texts = training_data['text'].tolist()
        labels = [label2id[label] for label in training_data['true_label']]
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
        val_dataset = IntentDataset(val_texts, val_labels, tokenizer)
        
        # Update model config for new labels
        model.config.num_labels = len(unique_labels)
        model.config.id2label = id2label
        model.config.label2id = label2id
        
        # Resize model embeddings if needed
        if model.config.num_labels != len(unique_labels):
            model.classifier = torch.nn.Linear(model.config.dim, len(unique_labels))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./retrained_model',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train the model
        trainer.train()
        
        # Save the retrained model
        model.save_pretrained('./retrained_model')
        tokenizer.save_pretrained('./retrained_model')
        
        print("Model retraining completed successfully!")
        
        # Update global model reference
        global model, tokenizer
        model = DistilBertForSequenceClassification.from_pretrained('./retrained_model')
        tokenizer = DistilBertTokenizerFast.from_pretrained('./retrained_model')
        model.eval()
        
        print("Retrained model loaded successfully!")
        
    except Exception as e:
        print(f"Error during retraining: {e}")

# --- Predict intent ---
@app.post("/predict")
def predict_intent_endpoint(message: Message):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check the dilbert_airline folder.")
    
    try:
        # Tokenize the input text
        inputs = tokenizer(message.text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Predict using DistilBERT
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
        
        # Get the predicted label using id2label mapping
        predicted_intent = model.config.id2label[predicted_class_id]
        
        return {"predicted_intent": predicted_intent, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Log feedback ---
@app.post("/feedback")
def log_feedback(feedback: Feedback):
    file_exists = os.path.isfile("feedback_log.csv")

    with open("feedback_log.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "predicted_intent", "correct", "true_label"])
        writer.writerow([
            feedback.text, 
            feedback.predicted_intent, 
            feedback.correct, 
            feedback.true_label
        ])

    # Check if retraining is needed
    if not feedback.correct:
        check_and_retrain_model()

    if feedback.correct:
        return {"message": "Fine. Glad I got it right!"}
    else:
        return {"message": "I'm transferring you to a human agent."}

# --- Manual retraining endpoint ---
@app.post("/retrain")
def manual_retrain():
    """Manually trigger model retraining"""
    try:
        if not os.path.exists("feedback_log.csv"):
            raise HTTPException(status_code=400, detail="No feedback data available")
        
        df = pd.read_csv("feedback_log.csv")
        retrain_model(df)
        
        return {"message": "Model retraining completed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
