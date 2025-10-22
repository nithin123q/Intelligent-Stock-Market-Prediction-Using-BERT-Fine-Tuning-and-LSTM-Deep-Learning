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




