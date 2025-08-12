# 📈 Financial News Sentiment Analysis with BERT

This project uses **BERT (Bidirectional Encoder Representations from Transformers)** to classify financial news headlines into **Negative**, **Neutral**, or **Positive** sentiments.
It includes both the **training pipeline** and a **saved pretrained model** so you can directly run predictions without retraining.

---

## 🚀 Features

* **BERT-based sentiment classification** using `bert-base-uncased`
* Handles **imbalanced classes** with class weights
* Supports **GPU acceleration** for faster training
* Ready-to-use **saved model** for instant predictions
* Modular scripts: training, preprocessing, prediction, and web app
* Training data augmented with **synonym replacement** to improve model robustness

---

## 📊 Dataset

The dataset used is the **Financial PhraseBank**, containing short financial news headlines labeled with sentiment.
Class distribution:

* **Neutral** → 59%
* **Positive** → 28%
* **Negative** → 12%

This imbalance is handled during training by computing **class weights**, ensuring that minority classes like *Negative* are given appropriate importance.
Additionally, data augmentation with **synonym replacement** was applied to increase diversity and help the model generalize better.

Example:

| Sentiment | Headline                                             |
| --------- | ---------------------------------------------------- |
| Positive  | Company profits soar in the latest quarterly report. |
| Neutral   | The central bank holds interest rates steady.        |
| Negative  | Shares plunge after weak earnings report.            |

---

## 📂 Project Structure

```
📦 sentiment-analysis-bert
 ┣ 📂 .git/                   # Git version control files
 ┣ 📂 __pycache__/            # Python cache files
 ┣ 📂 templates/              # HTML templates for web app
 ┣ 📜 all-data.xlsx           # Dataset (Financial PhraseBank)
 ┣ 📜 app.py                  # Flask app for web predictions
 ┣ 📜 config.json             # Model config
 ┣ 📜 financial Sentement analysis.ipynb # Jupyter notebook for experiments
 ┣ 📜 label_classes.txt       # Mapping of label IDs to sentiment names
 ┣ 📜 model.safetensors       # Trained BERT model weights
 ┣ 📜 predict.py              # Script to load model & run predictions
 ┣ 📜 preprocess.py           # Text preprocessing (if needed)
 ┣ 📜 special_tokens_map.json # Tokenizer special tokens
 ┣ 📜 tokenizer_config.json   # Tokenizer configuration
 ┣ 📜 train_model.py          # Script to train BERT model
 ┣ 📜 vocab.txt               # Tokenizer vocabulary
```

---

## 🛠 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/sentiment-analysis-bert.git
cd sentiment-analysis-bert
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare dataset**
   Place `all-data.xlsx` in the project folder.

---

## ▶️ Usage

### **Option 1: Train from scratch**

```bash
python train_model.py
```

This will:

* Load & preprocess the dataset
* Apply synonym-based data augmentation
* Train the BERT model
* Save the trained weights in `model.safetensors`

### **Option 2: Use the pre-trained model**

You already have `model.safetensors`, so you can run:

```bash
python predict.py
```

Example:

```
📝 Sentence: Company profits soar in the latest quarterly report.
🔎 Predicted Sentiment: Positive
```

---

## 🧠 Model Details

* **Base Model**: `bert-base-uncased` (HuggingFace Transformers)
* **Dropout**: 0.3 for regularization
* **Optimizer**: AdamW (lr=2e-5)
* **Loss Function**: CrossEntropyLoss with class weights
* **Batch Size**: 16
* **Epochs**: 3
* **Data Augmentation**: Synonym replacement for improved generalization

---

## 📈 Example Predictions

| Headline                                             | Predicted Sentiment |
| ---------------------------------------------------- | ------------------- |
| Company profits soar in the latest quarterly report. | Positive            |
| The central bank holds interest rates steady.        | Neutral             |
| Shares plunge after weak earnings report.            | Negative            |

---

## 🖥 Requirements

* Python 3.8+
* PyTorch
* Transformers
* scikit-learn
* tqdm
* pandas
* flask (if running the web app)
* openpyxl

Install them with:

```bash
pip install torch transformers scikit-learn tqdm pandas flask openpyxl
```

---

## 🌐 Running the Web App

If `app.py` is a Flask app:

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

to use the sentiment analysis web interface.

---

