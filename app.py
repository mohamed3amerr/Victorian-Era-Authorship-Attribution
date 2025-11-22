import os
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template
from transformers import AutoTokenizer

from src.model import AuthorClassifier
from src.data_preprocessing import preprocess_data

# --- Initialization ---
app = Flask(__name__)

# --- Global Variables ---
MODEL = None
TOKENIZER = None
AUTHOR_MAPPING = None
NUM_CLASSES = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model and Data Loading ---

def load_model_and_mapping():
    """
    Loads the fine-tuned model, tokenizer, and author mapping using the
    centralized `preprocess_data` function.
    """
    global MODEL, TOKENIZER, AUTHOR_MAPPING, NUM_CLASSES

    model_name = 'distilbert-base-uncased'
    model_path = 'models/distilbert-base-uncased_authors_best.pt'
    train_csv_path = 'dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv'

    # 1. Load mapping and num_classes from the official preprocessing script
    try:
        print("Loading author mapping using preprocess_data...")
        data = preprocess_data(train_csv=train_csv_path)
        AUTHOR_MAPPING = data.get('id_to_author')
        NUM_CLASSES = data.get('num_classes')

        if AUTHOR_MAPPING is None or NUM_CLASSES == 0:
            print("Error: Could not retrieve author mapping or number of classes.")
            return
        print(f"Successfully loaded mapping for {NUM_CLASSES} authors.")

    except Exception as e:
        print(f"An error occurred while generating author mapping: {e}")
        return

    # 2. Load Model and Tokenizer
    try:
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Prediction will be disabled.")
            return

        # Initialize the model architecture with the correct number of classes
        MODEL = AuthorClassifier(
            model_name=model_name,
            num_classes=NUM_CLASSES
        )

        # Load the saved state dictionary
        print("Loading model checkpoint...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.to(DEVICE)
        MODEL.eval()  # Set model to evaluation mode

        # Load the tokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)

        print(f"Model and tokenizer loaded successfully. Device: {DEVICE}")

    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        MODEL = None
        TOKENIZER = None

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    error_message = ""
    text_input = ""

    if request.method == 'POST':
        text_input = request.form.get('text', '').strip()

        if not text_input:
            error_message = "Please enter some text to predict."
        elif MODEL is None or TOKENIZER is None or AUTHOR_MAPPING is None:
            error_message = "Model is not available. Please ensure training is complete and the model file exists."
        else:
            try:
                # Prepare input for the model
                inputs = TOKENIZER(
                    text_input,
                    truncation=True,
                    padding='max_length',
                    max_length=256,  # Should match the training configuration
                    return_tensors='pt'
                )

                input_ids = inputs['input_ids'].to(DEVICE)
                attention_mask = inputs['attention_mask'].to(DEVICE)

                # Get prediction
                with torch.no_grad():
                    logits = MODEL(input_ids, attention_mask)
                    # Apply Softmax to get probabilities
                    probabilities = F.softmax(logits, dim=1)
                    
                    # Get top 3 predictions
                    top_probs, top_indices = torch.topk(probabilities, 3, dim=1)

                # Format results
                predictions = []
                for i in range(top_probs.size(1)):
                    prob = top_probs[0, i].item() * 100
                    idx = top_indices[0, i].item() # This is the model's internal 0-indexed prediction
                    author_label = AUTHOR_MAPPING.get(idx, "Unknown Author") # This is the author name/ID from the dataset
                    predictions.append({
                        "author_label": author_label,
                        "prediction_index": idx,
                        "probability": f"{prob:.2f}"
                    })

            except Exception as e:
                error_message = f"An error occurred during prediction: {e}"

    return render_template(
        'index.html',
        predictions=predictions,
        error=error_message,
        text_input=text_input
    )

if __name__ == '__main__':
    # Load the model and other necessary components when the app starts
    load_model_and_mapping()
    # Run the Flask app
    # Use 0.0.0.0 to make it accessible on the local network
    app.run(host='0.0.0.0', port=5000, debug=True)
