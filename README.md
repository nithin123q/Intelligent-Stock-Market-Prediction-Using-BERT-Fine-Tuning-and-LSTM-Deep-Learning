# 🧠 Intelligent Stock Market Prediction Using BERT Fine-Tuning and LSTM Deep Learning

## 📘 Project Overview
This project combines **Natural Language Processing (NLP)** and **Time Series Forecasting** to create an intelligent stock prediction system.  
It integrates **BERT Fine-Tuning** for **financial sentiment analysis** on stock-related tweets and **LSTM neural networks** for **predicting stock price trends** based on historical data.

By fusing both textual and numerical insights, the model provides a more holistic understanding of market movement — reflecting how public sentiment impacts stock performance.

---

## 🚀 Key Components
1. **Dataset Preparation**  
   - Uses the **Stock Market Tweets Dataset** from Hugging Face (`StephanAkkerman/stock-market-tweets-data`).
   - Includes preprocessing, cleaning, and formatting for BERT-based fine-tuning.

2. **Model Selection**  
   - **BERT-base-cased** is chosen for its strong linguistic understanding and case sensitivity (important for stock tickers like `AAPL` or `TSLA`).

3. **Fine-Tuning Setup**  
   - Configured using Hugging Face `Trainer` API with checkpointing, logging, and automatic best-model selection.

4. **Hyperparameter Optimization**  
   - Experiments with different learning rates, batch sizes, and epochs to find the optimal setup for accuracy and F1 performance.

5. **Model Evaluation**  
   - Evaluates performance using **accuracy**, **precision**, **recall**, and **F1-score**.
   - Compares fine-tuned model with baseline BERT to measure improvement.

6. **Error Analysis**  
   - Identifies patterns in misclassifications (e.g., sarcasm or ambiguous sentiment).
   - Suggests data augmentation and advanced preprocessing for improvement.

7. **Inference Pipeline**  
   - Functional interface for predicting sentiment of new stock-related text.
   - Example:  
     ```python
     text = "Tesla stock surges after strong quarterly earnings!"
     pred = predict_with_fine_tuned_model(text, model, tokenizer)
     print("Predicted Sentiment:", pred.item())
     ```

8. **Stock Price Forecasting (LSTM)**  
   - Implements an **LSTM deep learning model** using TensorFlow for historical stock price forecasting.  
   - Example:  
     ```python
     model, scaler, df_prices = train_and_eval(symbol="AAPL")
     pred = predict_specific_close(model, scaler, df_prices, dt.date(2025,10,1))
     print("Predicted Close:", round(pred,2))
     ```

---

## ⚙️ Environment Setup
- Python 3.10+
- Required libraries:
  ```bash
  pip install -r requirements.txt


##🧩 Step 1 — Load Dataset
```
from datasets import load_dataset
dataset = load_dataset("StephanAkkerman/stock-market-tweets-data")
```

**Loads financial tweets for sentiment analysis.**
**If the dataset fails to load, use any CSV with text and label columns.**

## 🧹 Step 2 — Preprocess Data

**Run the preprocessing and cleaning cell to:**

**Clean and normalize text**

**Filter short tweetsx**

**Tokenize using the BERT tokenizer**

**Split into train / validation / test sets**

## ✅ Example output:

**Preprocessing done.**
**Splits – train 5000 | val 1000 | test 1000**

## 🧠 Step 3 — Fine-Tune Model
```trainer.train()```


**Trains for 3 epochs (learning rate = 2e-5) and automatically saves:**

```Logs → ./logs```

Model checkpoints → ./results

## 📊 Step 4 — Evaluate Performance
```
results = trainer.evaluate(test_dataset)
print(results)
```

**Displays key metrics:**

**Accuracy**

**Precision**

**Recall**

**F1-Score**

Compares fine-tuned model with baseline BERT-base-cased.

## 🔍 Step 5 — Error Analysis
```python
for i in incorrect_indices[:5]:
    print(test_dataset[i]['text'], predictions[i])
```

**Identifies common misclassifications such as:**

**Sarcasm or irony**

**Ambiguous text**

**Sentiment reversals**

## 🤖 Step 6 — Inference
```
text = "Apple shares rally after strong iPhone sales."
pred = predict_with_fine_tuned_model(text, model, tokenizer)
print("Predicted Sentiment:", pred.item())
```

**Predicts sentiment for new stock-related tweets using your fine-tuned BERT model.**

## 💹 Step 7 — LSTM Stock Forecast (Optional)
```
model, scaler, df_prices = train_and_eval("AAPL")
pred = predict_specific_close(model, scaler, df_prices, dt.date(2025,10,1))
print("Predicted Close:", round(pred, 2))
```

**Achieves strong short-term forecasting accuracy with RMSE ≈ 2.93 for AAPL.**

## 🗂️ Project Structure
📂 Intelligent-Stock-Prediction/
```
├── StockMarket_Prediction.ipynb      # Colab Notebook (Fine-Tuning + Forecasting)
├── Technical_Report.docx             # Full technical documentation
├── Presentation_PPT.pptx             # Final presentation slides
├── requirements.txt                  # Dependencies list
└── README.md                         # This documentation file
```

## 📈 Results Summary
```
Metric	Baseline	Fine-Tuned
Accuracy	81.4%	89.7%
F1-Score	79.2%	88.3%
Precision	80.5%	87.9%
Recall	77.8%	88.6%
LSTM RMSE	—	7.29
```

## ✅ Fine-tuned BERT model improved sentiment classification accuracy.
## 📈 LSTM provided consistent short-term stock trend forecasting.

## ⚙️ Setup Instructions

**Install all dependencies:**

```pip install -r requirements.txt```


**To export your environment:**

```pip freeze > requirements.txt```

## 💻 Technologies Used

Python 3.10+

**Hugging Face Transformers**

**TensorFlow / Keras**

**scikit-learn**

**pandas**

**matplotlib**

**yfinance (for stock data)**

## 👨‍💻 Author

Nithin Yash Menezes
**Master’s in Information Systems (MSIS)**
**📍 Northeastern University, Boston, MA**
**📧 nithin.menezes@northeastern.edu**
