from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ✅ Load tokenizer and model from current directory
MODEL_DIR = "."  # Current directory where app.py and model files are located
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

# ✅ Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map.get(predicted_class_id, "Unknown")

# ✅ Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form["text"]
        prediction = predict_sentiment(input_text)
    return render_template("index.html", prediction=prediction, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
