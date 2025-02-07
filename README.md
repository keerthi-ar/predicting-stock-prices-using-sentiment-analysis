# Predicting Stock Prices Using Sentiment Analysis

## Objective:
The objective of this project is to predict stock prices by analyzing sentiment from financial news articles and social media posts. By combining sentiment analysis with historical stock data, we aim to forecast stock price movements.

## Dataset:
This project utilizes the **Financial News Dataset** from [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news). The dataset contains sentiment-labeled financial news articles, which will be used to predict stock price movements based on public sentiment.

## Project Overview:
In this project, we analyze the sentiment of financial news and social media posts to predict stock prices. We preprocess the text data using NLP techniques, perform sentiment analysis using Transformers (BERT), and combine sentiment scores with stock data to train a predictive model.

### Steps Taken:

#### 1. **Data Exploration and Preprocessing**:
- Explored and cleaned the dataset, handling any missing values or inconsistencies.
- Preprocessed text data by converting it to lowercase, removing stop words, and applying tokenization.

#### 2. **Sentiment Analysis**:
- Used **Transformers (BERT)** for sentiment analysis to classify the sentiment of financial news articles as **positive**, **neutral**, or **negative**.
- Mapped sentiment labels to numerical values for further use in modeling.

#### 3. **Stock Data Generation**:
- Simulated historical stock data (including open, high, low, close, and volume) to match the length of the sentiment data.
- Combined the sentiment scores with stock data to create a feature set for model training.

#### 4. **Model Building**:
- Used a **Gradient Boosting Regressor** to predict stock prices based on features including sentiment scores and historical stock data.
- Split the dataset into training and testing sets, evaluated the model's performance, and computed **Root Mean Squared Error (RMSE)**.

#### 5. **Model Evaluation**:
- Compared actual vs. predicted stock prices using scatter plots to visualize model performance.
- Integrated real-time sentiment data for future predictions and evaluated its impact on stock price forecasting.

#### 6. **Real-Time Sentiment Integration**:
- Incorporated new sentiment data (from recent news) to make predictions about stock prices based on current sentiment.

## Citation:
**Ankur Sinha, "Sentiment Analysis for Financial News," Kaggle, 2019.** [Online]. Available: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

## Installation:
To run this project on your local machine, you'll need to install the following libraries:

```bash
pip install transformers scikit-learn pandas numpy matplotlib seaborn
```

## Additional Dependencies:
The project was developed using Google Colab, so ensure you have the following libraries installed:

  - transformers: For sentiment analysis using pre-trained models (e.g., BERT).
  - scikit-learn: For model building and evaluation.
  - pandas and numpy: For data manipulation.
  - matplotlib and seaborn: For data visualization.

## Usage:

### Clone the repository:

```bash
git clone https://github.com/your-username/predicting-stock-prices-using-sentiment-analysis.git
```

### Navigate to the project directory:

```bash
cd predicting-stock-prices-using-sentiment-analysis
```

### Install the required libraries:

```bash
pip install -r requirements.txt
```

### Run the analysis:

```bash
jupyter notebook predicting_stock_prices_using_sentiment_analysis.ipynb
```
