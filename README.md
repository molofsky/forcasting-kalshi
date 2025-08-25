# Price-Pure Prediction of Daily Price Changes in Binary Event Contracts

# Overview 

This repository includes quantitative modeling techniques for price prediction for high-frequency trading, exploring whether the innately sentiment- and news-driven prices of binary event contracts can be accurately predicted with machine learning methods purely on the basis of timeprice data. This project benchmarks the performance of linear regression, decision trees, ensembles, artificial neural networks, and autoregressive integrated moving avergae methods on Kalshi binary event contracts.

# What is in this repo?
- `data`: Contains dataset processing, feature engineering, and dataloaders for feeding training data to models
- `models`: Contains PyTorch models, such as ensembles, artifical neural networks, and autoregressive integrated moving average methods,for predicting daily price changes

# Dataset

This dataset derives from 10,065 examples of High-volume binary event contracts spanning all topics, years, resolved, and active collected from Kalshi. The rows are daily examples, columns features. The target output is one-day-ahead percentage price change. There are 18 price/date-derived features. Features of low importance in influencing the target are pruned to optimize model performance. The models are trained using k-fold cross validation to combat sampling bias.

# Running models

1. Create a Python virtual environment

   ```
   python3 -m venv forecasting-kalshi
   source forecasting-kalshi/bin/activate
   ```
   
2. Install Python dependencies

   ```
   pip install -r requirements.txt
   ```

3. Start training
   ```
   python models.py
   ```

# Contributing

Please format the code before submitting changes.

   ```
   python -m black *.py
   ```
