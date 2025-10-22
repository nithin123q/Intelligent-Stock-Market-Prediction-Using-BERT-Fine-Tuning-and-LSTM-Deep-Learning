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


# 🧠 Intelligent Stock Market Prediction Using BERT Fine-Tuning and LSTM Deep Learning
#
# This project combines BERT fine-tuning for financial tweet sentiment analysis
# with an LSTM-based deep learning model for time-series stock forecasting.
# It demonstrates an end-to-end intelligent system that interprets market sentiment
# and predicts future price trends.

# --------------------------------------------------------
# 🚀 Quick Start (Colab Link)
# --------------------------------------------------------
# Open the project in Google Colab:
# 🔗 https://colab.research.google.com/drive/18JuWsYk2D8PXxjHdu9MzTotFufubExUk?usp=sharing

# --------------------------------------------------------
# 🧩 Step 1 — Load Dataset
# --------------------------------------------------------
# from datasets import load_dataset
# dataset = load_dataset("StephanAkkerman/stock-market-tweets-data")
#
# This loads financial tweets for sentiment analysis.
# If the dataset fails to load, replace it with any CSV containing `text` and `label` columns.

# --------------------------------------------------------
# 🧹 Step 2 — Preprocess Data
# --------------------------------------------------------
# Run the “Preprocessing and Cleaning” cell to:
# - Clean and normalize text
# - Filter short tweets
# - Tokenize using the BERT tokenizer
# - Split into train / validation / test sets
#
# Example output:
# ✅ Preprocessing done.
# Splits – train 5000 | val 1000 | test 1000

# --------------------------------------------------------
# 🧩 Step 3 — Fine-Tune the Model
# --------------------------------------------------------
# trainer.train()
#
# Trains for 3 epochs (learning rate = 2e-5) and automatically saves:
# - Training logs → ./logs
# - Model checkpoints → ./results

# --------------------------------------------------------
# 📈 Step 4 — Evaluate Performance
# --------------------------------------------------------
# results = trainer.evaluate(test_dataset)
# print(results)
#
# Displays key metrics:
# - Accuracy
# - Precision
# - Recall
# - F1-Score
#
# and compares them against the baseline (pre-fine-tuned BERT-base-cased).

# --------------------------------------------------------
# 🔍 Step 5 — Run Error Analysis
# --------------------------------------------------------
# # Displays incorrect predictions
# for i in incorrect_indices[:5]:
#     print(test_dataset[i]['text'], predictions[i])
#
# Helps identify common misclassifications such as:
# - Sarcasm or irony
# - Ambiguous phrasing
# - Sentiment reversal (e.g., “down only slightly” misread as positive)

# --------------------------------------------------------
# 🤖 Step 6 — Perform Inference
# --------------------------------------------------------
# text = "Apple shares rally after strong iPhone sales."
# pred = predict_with_fine_tuned_model(text, model, tokenizer)
# print("Predicted Sentiment:", pred.item())
#
# Predicts sentiment for new stock-related text using your fine-tuned BERT model.

# --------------------------------------------------------
# 💹 Step 7 — Forecast Stock Prices (Optional)
# --------------------------------------------------------
# model, scaler, df_prices = train_and_eval("AAPL")
# pred = predict_specific_close(model, scaler, df_prices, dt.date(2025,10,1))
# print("Predicted Close:", round(pred,2))
#
# Achieves reliable short-term forecasting performance using LSTM.

# --------------------------------------------------------
# 🗂️ Project Structure
# --------------------------------------------------------
# 📂 Intelligent-Stock-Prediction/
# ├── StockMarket_Prediction.ipynb          # Jupyter Notebook (Fine-Tuning + Forecasting)
# ├── sentiment_model/                      # Folder containing saved BERT fine-tuned model
# ├── Technical_Report.docx                 # Full technical documentation
# ├── Presentation_PPT.pptx                 # Final presentation slides
# ├── requirements.txt                      # Dependencies list
# ├── README.md                             # Documentation file (this file)

# --------------------------------------------------------
# 📊 Results Summary
# --------------------------------------------------------
# Metric        | Baseline BERT | Fine-Tuned BERT
# --------------|---------------|----------------
# Accuracy      | 81.4%         | 89.7%
# F1-Score      | 79.2%         | 88.3%
# Precision     | 80.5%         | 87.9%
# Recall        | 77.8%         | 88.6%
#
# 📉 LSTM forecasting achieved an RMSE of 2.93,
# demonstrating reliable short-term trend prediction for AAPL stock.

# --------------------------------------------------------
# 🧠 Technologies Used
# --------------------------------------------------------
# - Python 3.10+
# - Hugging Face Transformers
# - TensorFlow / Keras
# - scikit-learn, pandas, matplotlib
# - yfinance (for stock data)

# --------------------------------------------------------
# ⚙️ Setup
# --------------------------------------------------------
# To install all dependencies:
# pip install -r requirements.txt
#
# To export your current environment:
# pip freeze > requirements.txt

# --------------------------------------------------------
# 👨‍💻 Author
# --------------------------------------------------------
# Nithin Yash Menezes
# 📧 nithin.menezes@northeastern.edu
# 📍 Northeastern University, Boston MA
# 💼 Master’s in Information Systems (MSIS)

# --------------------------------------------------------
# 🏁 License
# --------------------------------------------------------
# This project is open-source and intended for educational and research purposes only.

Recall	77.8%	88.6%

LSTM forecasting achieved an RMSE of 2.93, demonstrating reliable short-term trend prediction for AAPL stock.


