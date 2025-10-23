# Airline Chatbot with DistilBERT Model

A customer service chatbot for airlines that uses a pre-trained DistilBERT model for intent classification, running completely locally without external API dependencies.

## Features

- **Pre-trained DistilBERT Model**: Uses a fine-tuned DistilBERT transformer model for superior accuracy
- **28 Intent Categories**: Recognizes comprehensive airline-related intents including booking, cancellation, baggage, seat selection, and more
- **Feedback Loop**: Collects user feedback to improve model accuracy
- **Human Handoff**: Escalates to human agents when AI predictions are incorrect
- **Local Processing**: All inference happens locally without external API calls

## Setup

### Prerequisites
- Python 3.7+
- Node.js 14+
- The `dilbert_airline` folder with pre-trained model files

### Quick Setup
Run the setup script to install dependencies and verify the model:

```bash
python setup.py
```

### Manual Setup

1. **Backend Setup**:
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py         # Start the FastAPI server
   ```

2. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

## Usage

1. Open your browser to `http://localhost:3000`
2. Type airline-related queries like:
   - "I want to book a flight"
   - "Cancel my reservation"
   - "What's my flight status?"
   - "I need help with baggage"
3. The chatbot will predict the intent and ask for feedback
4. Mark predictions as correct ✅ or incorrect ❌

## Supported Intents

The DistilBERT model recognizes 28 comprehensive airline intents:

**Core Flight Operations:**
- Cancel Trip, Change Flight, Flight Status, Flights Info

**Baggage & Luggage:**
- Check In Luggage Faq, Carry On Luggage Faq, Missing Bag, Baggage Delay, Damaged Bag

**Seat & Upgrade Services:**
- Seat Availability, Seat Upgrade Request

**Travel Services:**
- Airport Lounge Access, Airport Transfers, Special Assistance, Pet Travel

**Policies & Support:**
- Cancellation Policy, Insurance, Medical Policy, Travel Documents, Prohibited Items Faq

**Customer Service:**
- Complaints, Refund / Compensation, Frequent Flyer, Discounts, Fare Check

**Additional Services:**
- Meal Preferences, Sports Music Gear, Travel Alerts

## Model Information

This chatbot uses a pre-trained DistilBERT model that has been fine-tuned specifically for airline customer service intents. The model:

- **Architecture**: DistilBERT (distilled version of BERT)
- **Training**: Fine-tuned on airline customer service data
- **Performance**: High accuracy on 28 intent categories
- **Size**: Optimized for local deployment
- **Location**: Stored in the `dilbert_airline` folder

## Feedback Collection

User feedback is logged to `feedback_log.csv` with columns:
- `text`: User's original message
- `predicted_intent`: What the model predicted
- `correct`: Whether the prediction was correct

## Architecture

- **Backend**: FastAPI with DistilBERT transformer model
- **Frontend**: React.js with chat interface
- **ML Model**: Pre-trained DistilBERT for sequence classification
- **Data**: CSV logging for feedback collection

## Customization

### Model Retraining
Since this uses a pre-trained DistilBERT model, retraining would require:
1. Access to the original training dataset
2. Fine-tuning the model with additional airline data
3. Saving the updated model to the `dilbert_airline` folder

### Adding New Features
1. **New Intents**: Would require model retraining with labeled data
2. **UI Improvements**: Modify `frontend/src/App.js`
3. **Backend Logic**: Extend `backend/main.py` with new endpoints

## Troubleshooting

- **Model not found**: Ensure the `dilbert_airline` folder exists in the project root
- **Import errors**: Install requirements with `pip install -r requirements.txt`
- **CUDA/GPU issues**: The model runs on CPU by default; GPU support available with CUDA
- **CORS issues**: The backend is configured to allow all origins for development
- **Port conflicts**: Backend runs on port 8000, frontend on port 3000
- **Memory issues**: DistilBERT requires ~500MB RAM; ensure sufficient system memory
